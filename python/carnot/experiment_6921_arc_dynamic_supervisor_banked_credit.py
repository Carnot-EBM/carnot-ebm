"""Audit live supervisor receipts with durable banked-progress credit.

REQ-ARC-WMTE-6921 fixes two evidence defects without running ARC. The audit
finds receipt content under declared run directories instead of naming one
old artifact. It then replaces transient supervisor credit with level changes
that the finished row says were banked. The existing refinement policy reads
an in-memory corrected ledger and remains recommendation-only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.agentic.arc_solve_artifact_discipline import (
    ARC_DYNAMIC_SUPERVISOR_BANKED_CREDIT_SUBSTRATE,
)
from carnot.agentic.arc_supervisor_refinement import (
    LEDGER_SCHEMA,
    MAX_UNREDIRECTED_WINDOWS,
    MIN_FIRED_PER_ARM,
    classify_receipt,
    empty_ledger,
    evaluate,
    receipt_id_for_row,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
OUTPUT_PATH = Path("results/experiment_6921_arc_dynamic_supervisor_banked_credit.json")
# REQ-ARC-WMTE-6642: the eval-run fields this module requires. Checked by
# scripts/eval_run_consumer_field_lint.py against real artifacts + producer source.
EVAL_RUN_FIELDS_READ = ("per_game", "game", "complete", "honest_verdict", "trajectory_supervisor")
LEDGER_PATH = Path("ops/arc_supervisor_refinement_ledger.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
SCORED_PATH_RUN_DIRECTORY = Path("results/arc_leaderboard_eval_runs")
CANONICAL_ENTRYPOINTS = (
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("scripts/arc_loop_solve.py"),
)
PRIOR_ARTIFACT_PATHS = {
    "experiment_6681": Path("results/experiment_6681_arc_post_redirect_outcomes.json"),
    "experiment_6844": Path("results/experiment_6844_supervisor_action_outcome_credit_audit.json"),
    "experiment_6857": Path("results/experiment_6857_dynamic_live_arc_receipt_router.json"),
    "experiment_6858": Path("results/experiment_6858_supervisor_counterfactual_credit_audit.json"),
}
EXPECTED_PRIOR_ARTIFACT_HASHES = {
    "experiment_6681": "sha256:bf61e50970530a9844103e59f702c68e32a8aba12f78b1c68dd6154685b9107a",
    "experiment_6844": "sha256:b19a7981b9d7e587f42ad3be42cabeda45aff08046a3064c90396f1502cf1b71",
    "experiment_6857": "sha256:822121d5106b7a9f019dd3b092fec607643211f414e2dfecfa9bf98ca08f321f",
    "experiment_6858": "sha256:6e7b8a9f9da68b96e98ba23be94ed5a38572d573de6ccc4f1535602a44386d9a",
}
SCHEMA = "carnot.experiment_6921.arc_dynamic_supervisor_banked_credit.v1"
INFERENCE_SUBSTRATE = ARC_DYNAMIC_SUPERVISOR_BANKED_CREDIT_SUBSTRATE
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "receipt_discovery_manifest",
    "rows",
    "discovered_file_rows",
    "provenance_rows",
    "dedupe_rows",
    "applied_receipt_rows",
    "shadow_receipt_rows",
    "error_receipt_rows",
    "redirect_rows",
    "banked_level_transition_rows",
    "actions_to_progress_rows",
    "censored_rows",
    "competing_redirect_rows",
    "transient_vs_banked_credit_rows",
    "per_game_rows",
    "per_arm_rows",
    "refinement_recommendation_rows",
    "automatic_arm_mutation_count",
    "new_eligible_receipt_count",
    "banked_progress_event_count",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "arc_supervisor_audit_complete_score",
    "banked_credit_eligible_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "The schema lets later reducers reject incompatible audit documents.",
    "experiment_id": "The identifier binds this result to REQ-ARC-WMTE-6921.",
    "run_date": "The date identifies the requested deterministic audit execution.",
    "status": "The status separates completed audit work from evidence eligibility.",
    "field_principles": "Each artifact field states why it must remain in later reductions.",
    "preconditions_checked": "Named checks stop discovery when required evidence is unavailable or changed.",
    "inference_substrate": "The substrate states that this audit uses cached live receipts and no new model call.",
    "duration_s": "Measured wall time exposes a skipped or fabricated reducer execution.",
    "source_artifact_hashes": "Exact hashes bind every prior audit and owned input used here.",
    "receipt_discovery_manifest": "Declared roots and counts make dynamic discovery reproducible.",
    "rows": "One row per redirect keeps old credit, banked credit, and disposition together.",
    "discovered_file_rows": "File rows prove discovery used content across all readable run files.",
    "provenance_rows": "Provenance keeps live-agent evidence separate from proxies and partial runs.",
    "dedupe_rows": "Canonical identities prevent copied rows and partial-final copies from inflating support.",
    "applied_receipt_rows": "Only applied live receipts can enter effect evidence.",
    "shadow_receipt_rows": "Shadow rows remain in the denominator but represent actions never applied.",
    "error_receipt_rows": "Error markers remain visible and cannot become effect evidence.",
    "redirect_rows": "Redirect rows expose the complete replay decision for each applied redirect.",
    "banked_level_transition_rows": "Durable transitions are the only allowed source of progress credit.",
    "actions_to_progress_rows": "Recomputed action gaps expose stale or incorrect old timing fields.",
    "censored_rows": "Censoring distinguishes no later banked event from a measured positive.",
    "competing_redirect_rows": "Shared later transitions expose credit competition and prevent causal wording.",
    "transient_vs_banked_credit_rows": "Side-by-side credit shows when transient progress inflated old counts.",
    "per_game_rows": "Game-run summaries retain existing outcomes without claiming a new solve.",
    "per_arm_rows": "The frozen policy support table shows which arms meet its evidence floor.",
    "refinement_recommendation_rows": "Recommendations remain human-only outputs of the existing policy.",
    "refinement_policy": "The complete policy receipt proves its contract stayed recommendation-only.",
    "refinement_input_receipt_count": "The input count proves shadow, error, and rejected rows did not enter policy evidence.",
    "automatic_arm_mutation_count": "Zero protects the curated arm table from automatic changes.",
    "new_eligible_receipt_count": "The count separates a fresh audit from a replay of the durable ledger.",
    "banked_progress_event_count": "Nonzero durable progress is required before effect eligibility.",
    "solve_provenance": "Live self-discovery provenance prevents proxy outcomes from becoming headline evidence.",
    "random_seed": "A fixed seed identifies deterministic tie and checksum behavior.",
    "reproducibility_checksum": "One digest binds all stable artifact content except itself and duration.",
    "arc_supervisor_audit_complete_score": "This score covers discovery, provenance, dedupe, and replay, not effect.",
    "banked_credit_eligible_score": "This score requires fresh rows, banked headroom, and the frozen evidence floor.",
    "gate_check_summary": "A blocked or null result names the first failed check and exact values.",
    "verifier_is_oracle": "False states that this reducer audits external receipts and is not the game oracle.",
    "verdict_class": "A closed class prevents an evidence audit from becoming an unsupported positive.",
    "honest_verdict": "The complete_ prefix marks the audit as terminal at its measured boundary.",
    "solve_claim": "False prevents existing level outcomes from becoming a new solve claim.",
    "new_solve_count": "Zero states that this audit did not bank or register a new level.",
    "arc_run_launch_count": "Zero proves the task consumed existing rows instead of launching ARC.",
    "game_adapter_mutation_count": "Zero protects public-game adapters from a generalization audit.",
    "registry_mutation_count": "Zero protects the solve registry from read-only evidence work.",
    "source_game_files_read": "An empty list excludes hidden or public game-source reverse engineering.",
}


def canonical_bytes(value: Any) -> bytes:
    """Encode stable JSON bytes so content identity ignores formatting."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 value used by repository artifacts."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading a large receipt into a second buffer."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_row_id(row: Mapping[str, Any]) -> str:
    """Use the standing ledger's full-row identity for copy dedupe."""

    return receipt_id_for_row(dict(row))


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable content while excluding duration and the digest itself."""

    copy = deepcopy(dict(artifact))
    copy.pop("duration_s", None)
    copy.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_bytes(copy))


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _read_yaml(path: Path) -> tuple[Any, str | None]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_check": failed[0]["check"] if failed else None,
        "expected": failed[0]["expected"] if failed else "all checks pass",
        "observed": failed[0]["observed"] if failed else "all checks pass",
        "failed_checks": failed,
        "checks": [dict(row) for row in checks],
    }


def _git_head(root: Path) -> str:
    """Read the checkout identity without changing or refreshing git state."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unrecorded_source_commit"
    value = result.stdout.strip()
    return value if value else "unrecorded_source_commit"


def _default_discovery_roots(root: Path, ledger: Mapping[str, Any]) -> list[JsonDict]:
    """Find the lever root from durable source provenance, not a user path."""

    roots: list[JsonDict] = [{"kind": "scored_path", "path": SCORED_PATH_RUN_DIRECTORY.as_posix()}]
    source_parents = sorted(
        {
            str(Path(entry["source"]).expanduser().resolve().parent)
            for entry in ledger.get("entries", {}).values()
            if isinstance(entry, Mapping) and isinstance(entry.get("source"), str)
        }
    )
    if source_parents:
        common = Path(os.path.commonpath(source_parents))
        roots.append({"kind": "lever_harness", "path": str(common)})
    else:
        roots.append({"kind": "lever_harness", "path": "missing_ledger_source_directory"})
    return roots


def _resolve_root(root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _source_hash_row(name: str, path: Path, root: Path) -> JsonDict:
    exists = path.is_file()
    return {
        "name": name,
        "path": _relative_or_absolute(path, root),
        "exists": exists,
        "file_sha256": sha256_file(path) if exists else None,
    }


def precondition_checks(
    root: Path,
    discovery_roots: Sequence[Mapping[str, Any]],
    expected_prior_hashes: Mapping[str, str],
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Check every owned resource before scanning any receipt content."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in CANONICAL_ENTRYPOINTS:
        path = root / relative
        hashes[relative.as_posix()] = _source_hash_row(relative.as_posix(), path, root)
        available = path.is_file() and os.access(path, os.R_OK)
        checks.append(
            _check(f"canonical_entrypoint:{relative.as_posix()}", True, available, available)
        )

    ledger_path = root / LEDGER_PATH
    ledger, ledger_error = _read_json(ledger_path)
    ledger_ok = (
        isinstance(ledger, dict)
        and ledger.get("schema") == LEDGER_SCHEMA
        and isinstance(ledger.get("entries"), dict)
    )
    hashes[LEDGER_PATH.as_posix()] = _source_hash_row("durable_ledger", ledger_path, root)
    checks.append(
        _check("durable_refinement_ledger_readable", True, ledger_error or ledger_ok, ledger_ok)
    )

    registry_path = root / REGISTRY_PATH
    registry, registry_error = _read_yaml(registry_path)
    registry_ok = isinstance(registry, dict) and isinstance(registry.get("games"), list)
    hashes[REGISTRY_PATH.as_posix()] = _source_hash_row("current_registry", registry_path, root)
    checks.append(
        _check("current_registry_readable", True, registry_error or registry_ok, registry_ok)
    )

    for index, root_row in enumerate(discovery_roots):
        path = _resolve_root(root, str(root_row.get("path", "")))
        available = path.is_dir() and os.access(path, os.R_OK)
        checks.append(
            _check(
                f"run_directory_readable:{root_row.get('kind', index)}",
                True,
                available,
                available,
            )
        )

    for name, relative in PRIOR_ARTIFACT_PATHS.items():
        path = root / relative
        row = _source_hash_row(name, path, root)
        hashes[relative.as_posix()] = row
        expected = expected_prior_hashes.get(name)
        observed = row["file_sha256"]
        checks.append(
            _check(f"prior_artifact_hash:{name}", expected, observed, observed == expected)
        )
    safe_ledger = ledger if ledger_ok else empty_ledger()
    return checks, hashes, safe_ledger


def _walk_json_files(root: Path) -> tuple[list[Path], list[str]]:
    """Walk JSON files and stop at each nested repository boundary."""

    files: list[Path] = []
    pruned: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames or ".git" in filenames:
            pruned.append(str(Path(dirpath).resolve()))
            dirnames[:] = []
            continue
        for filename in filenames:
            if filename.lower().endswith(".json"):
                files.append(Path(dirpath) / filename)
    return sorted(files, key=str), sorted(pruned)


def _source_rows(payload: Any) -> tuple[list[JsonDict], str | None]:
    if isinstance(payload, dict):
        for key, shape in (("per_game", "scored_path_per_game"), ("rows", "lever_rows")):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)], shape
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)], "bare_rows"
    return [], None


def _document_terminal(payload: Any, shape: str | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    verdict = str(payload.get("honest_verdict") or "")
    if payload.get("complete") is False or verdict.startswith("partial_"):
        return False
    if shape == "scored_path_per_game":
        return payload.get("complete") is True or verdict.startswith(("complete_", "complete:"))
    return True


def _run_id(payload: Any, path: Path) -> str:
    if isinstance(payload, Mapping):
        for key in ("run_id", "attempt_identity", "episode_id"):
            if payload.get(key):
                return str(payload[key])
    name = path.name
    if name.endswith(".partial.json"):
        name = name[: -len(".partial.json")]
    elif name.endswith(".json"):
        name = name[: -len(".json")]
    return name


def discover_receipts(
    root: Path,
    discovery_roots: Sequence[Mapping[str, Any]],
    *,
    source_commit_fallback: str,
) -> JsonDict:
    """Discover receipt rows from content and retain file-level provenance."""

    discovered_files: list[JsonDict] = []
    candidates: list[JsonDict] = []
    manifest_roots: list[JsonDict] = []
    seen_files: set[Path] = set()
    for root_row in discovery_roots:
        run_root = _resolve_root(root, str(root_row["path"]))
        files, pruned = _walk_json_files(run_root)
        receipt_file_count = 0
        for path in files:
            resolved = path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            payload, load_error = _read_json(path)
            source_rows, shape = _source_rows(payload)
            receipt_rows = [
                row for row in source_rows if isinstance(row.get("trajectory_supervisor"), dict)
            ]
            if not receipt_rows:
                continue
            receipt_file_count += 1
            file_hash = sha256_file(path)
            terminal = _document_terminal(payload, shape)
            run_id = _run_id(payload, path)
            declared_commit = payload.get("source_commit") if isinstance(payload, Mapping) else None
            source_commit = str(declared_commit or source_commit_fallback)
            source_commit_basis = (
                "document" if declared_commit else "discovery_repository_head_fallback"
            )
            file_row = {
                "path": _relative_or_absolute(path, root),
                "content_sha256": file_hash,
                "directory_kind": str(root_row.get("kind")),
                "content_shape": shape,
                "run_id": run_id,
                "source_commit": source_commit,
                "source_commit_basis": source_commit_basis,
                "terminal": terminal,
                "receipt_row_count": len(receipt_rows),
                "load_error": load_error,
            }
            discovered_files.append(file_row)
            for source_index, row in enumerate(receipt_rows):
                candidates.append(
                    {
                        "row": row,
                        "source_index": source_index,
                        **file_row,
                    }
                )
        manifest_roots.append(
            {
                "kind": str(root_row.get("kind")),
                "path": _relative_or_absolute(run_root, root),
                "json_file_count": len(files),
                "receipt_file_count": receipt_file_count,
                "nested_repository_roots_pruned": pruned,
            }
        )
    return {
        "files": sorted(discovered_files, key=lambda row: row["path"]),
        "candidates": candidates,
        "roots": manifest_roots,
    }


def _provenance_disposition(candidate: Mapping[str, Any]) -> tuple[str, str]:
    row = candidate["row"]
    if candidate.get("load_error"):
        return "rejected", "unreadable_json"
    if candidate.get("terminal") is not True:
        return "rejected", "nonterminal_receipt_file"
    provenance = str(row.get("solve_provenance") or "live_agent_self_discovery")
    if provenance != "live_agent_self_discovery":
        return "rejected", provenance
    if row.get("read_game_source") is True or row.get("used_env_source") is True:
        return "rejected", "source_reading"
    if row.get("outer_loop_re") is True:
        return "rejected", "outer_loop_re"
    if row.get("llm_on_row_valid") is False:
        return "rejected", "invalid_live_harness_row"
    kind = classify_receipt(dict(row))
    if kind == "applied":
        return "applied", "eligible_live_applied_receipt"
    if kind == "shadow":
        return "shadow", "counterfactual_not_applied"
    if kind == "error":
        return "error", "receipt_error_marker"
    return "rejected", f"receipt_{kind}"


def qualify_and_dedupe(candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Choose one best copy of each canonical row and preserve all copies."""

    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for candidate in candidates:
        row_id = canonical_row_id(candidate["row"])
        disposition, reason = _provenance_disposition(candidate)
        grouped[row_id].append(
            {
                **dict(candidate),
                "canonical_row_id": row_id,
                "disposition": disposition,
                "reason": reason,
            }
        )

    rank = {"applied": 0, "shadow": 1, "error": 2, "rejected": 3}
    selected: list[JsonDict] = []
    provenance_rows: list[JsonDict] = []
    dedupe_rows: list[JsonDict] = []
    for row_id, copies in sorted(grouped.items()):
        ordered = sorted(copies, key=lambda row: (rank[row["disposition"]], row["path"]))
        chosen = ordered[0]
        selected.append(chosen)
        for index, copy in enumerate(ordered):
            provenance_rows.append(
                {
                    "path": copy["path"],
                    "content_sha256": copy["content_sha256"],
                    "run_id": copy["run_id"],
                    "game": copy["row"].get("game"),
                    "arm": copy["row"].get("arm"),
                    "receipt_mode": classify_receipt(copy["row"]),
                    "source_commit": copy["source_commit"],
                    "source_commit_basis": copy["source_commit_basis"],
                    "canonical_row_id": row_id,
                    "solve_provenance": str(
                        copy["row"].get("solve_provenance") or "live_agent_self_discovery"
                    ),
                    "disposition": copy["disposition"],
                    "reason": copy["reason"],
                    "selected_copy": index == 0,
                }
            )
            dedupe_rows.append(
                {
                    "canonical_row_id": row_id,
                    "path": copy["path"],
                    "selected_path": chosen["path"],
                    "disposition": "unique"
                    if len(ordered) == 1
                    else ("selected_copy" if index == 0 else "duplicate_copy"),
                }
            )
    return {
        "selected": selected,
        "provenance_rows": provenance_rows,
        "dedupe_rows": dedupe_rows,
    }


def _receipt_projection(candidate: Mapping[str, Any]) -> JsonDict:
    row = candidate["row"]
    return {
        "path": candidate["path"],
        "content_sha256": candidate["content_sha256"],
        "run_id": candidate["run_id"],
        "game": row.get("game"),
        "arm": row.get("arm"),
        "receipt_mode": classify_receipt(row),
        "source_commit": candidate["source_commit"],
        "canonical_row_id": candidate["canonical_row_id"],
        "solve_provenance": str(row.get("solve_provenance") or "live_agent_self_discovery"),
        "disposition": candidate["reason"],
    }


def _banked_transitions(candidate: Mapping[str, Any]) -> list[JsonDict]:
    """Project only transitions included in the row's durable level total."""

    row = candidate["row"]
    levels = row.get("levels", 0)
    durable_levels = int(levels) if isinstance(levels, int) and not isinstance(levels, bool) else 0
    raw_actions = row.get("level_up_charged")
    actions = (
        [value for value in raw_actions if isinstance(value, int)]
        if isinstance(raw_actions, list)
        else []
    )
    if len(actions) < durable_levels:
        for frame in row.get("frame_sequence", []):
            if not isinstance(frame, Mapping):
                continue
            level_after = frame.get("levels_completed")
            if not isinstance(level_after, int) or level_after <= len(actions):
                continue
            action = frame.get("charged_action_index")
            if not isinstance(action, int):
                index = frame.get("frame_index")
                action = index + 1 if isinstance(index, int) else None
            if isinstance(action, int):
                actions.append(action)
            if len(actions) >= durable_levels:
                break
    transitions: list[JsonDict] = []
    for index, action in enumerate(actions[:durable_levels]):
        material = {
            "canonical_row_id": candidate["canonical_row_id"],
            "level_after": index + 1,
            "action_index": action,
        }
        transitions.append(
            {
                "transition_id": sha256_bytes(canonical_bytes(material)),
                "canonical_row_id": candidate["canonical_row_id"],
                "run_id": candidate["run_id"],
                "game": row.get("game"),
                "level_before": index,
                "level_after": index + 1,
                "action_index": action,
                "banked": True,
                "solve_provenance": str(row.get("solve_provenance") or "live_agent_self_discovery"),
            }
        )
    return sorted(transitions, key=lambda item: (item["action_index"], item["level_after"]))


def replay_banked_credit(applied: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay each redirect against its own strictly later banked transition."""

    transitions: list[JsonDict] = []
    redirects: list[JsonDict] = []
    corrected_entries: dict[str, JsonDict] = {}
    per_game: list[JsonDict] = []
    for candidate in applied:
        row = candidate["row"]
        receipt = row["trajectory_supervisor"]
        row_transitions = _banked_transitions(candidate)
        transitions.extend(row_transitions)
        corrected_redirects: list[JsonDict] = []
        row_redirects: list[JsonDict] = []
        for source_index, redirect in enumerate(receipt.get("redirects", [])):
            if not isinstance(redirect, Mapping) or redirect.get("arm") is None:
                continue
            action_index = redirect.get("action_index")
            later = (
                next(
                    (
                        event
                        for event in row_transitions
                        if isinstance(action_index, int) and event["action_index"] > action_index
                    ),
                    None,
                )
                if isinstance(action_index, int)
                else None
            )
            actions_to_banked = later["action_index"] - action_index if later is not None else None
            old_actions = redirect.get("actions_to_levelup")
            old_credit = redirect.get("resolved_by_levelup") is True
            redirect_id = sha256_bytes(
                canonical_bytes(
                    {
                        "canonical_row_id": candidate["canonical_row_id"],
                        "source_index": source_index,
                        "redirect": redirect,
                    }
                )
            )
            replay = {
                "redirect_id": redirect_id,
                "canonical_row_id": candidate["canonical_row_id"],
                "run_id": candidate["run_id"],
                "game": row.get("game"),
                "harness_arm": row.get("arm"),
                "redirect_arm": str(redirect.get("arm")),
                "redirect_action_index": action_index,
                "redirect_level": redirect.get("level"),
                "old_credit": old_credit,
                "old_helped_count": (
                    receipt.get("arm_outcomes", {}).get(str(redirect.get("arm")), {}).get("helped")
                    if isinstance(receipt.get("arm_outcomes"), Mapping)
                    else None
                ),
                "old_actions_to_progress": old_actions,
                "banked_credit": later is not None,
                "banked_transition_id": later["transition_id"] if later else None,
                "banked_transition_action_index": later["action_index"] if later else None,
                "actions_to_banked_progress": actions_to_banked,
                "actions_to_progress_mismatch": old_actions != actions_to_banked,
                "censored": later is None,
                "no_progress_outcome": later is None,
                "disposition": "banked_progress" if later else "censored_no_banked_progress",
                "competing_redirect_ids": [],
                "competing_redirect_count": 0,
                "solve_provenance": str(row.get("solve_provenance") or "live_agent_self_discovery"),
            }
            redirects.append(replay)
            row_redirects.append(replay)
            corrected_redirects.append(
                {
                    "arm": str(redirect.get("arm")),
                    "action_index": action_index,
                    "level": redirect.get("level"),
                    "resolved_by_levelup": later is not None,
                    "actions_to_levelup": actions_to_banked,
                }
            )
        row_id = candidate["canonical_row_id"]
        corrected_entries[row_id] = {
            "source": candidate["path"],
            "game": row.get("game"),
            "seed": row.get("seed"),
            "harness_arm": row.get("arm"),
            "window": receipt.get("window"),
            "mode": "applied",
            "actions_observed": receipt.get("actions_observed", row.get("charged_actions")),
            "stagnations_unredirected": int(receipt.get("stagnations_unredirected") or 0),
            "levels": row.get("levels"),
            "redirects": corrected_redirects,
            "receipt_id": row_id,
            # REQ-ARC-WMTE-7033: exhaustion is read per level from the receipt's own window
            # rows. Without them the corrected ledger could only pool arms across the run,
            # which is the defect the refinement tool no longer commits.
            "arms_enabled": (
                [str(arm) for arm in receipt["arms_enabled"]]
                if isinstance(receipt.get("arms_enabled"), list)
                else None
            ),
            "unredirected_windows": [
                dict(item)
                for item in (receipt.get("unredirected_windows") or [])
                if isinstance(item, Mapping)
            ][:MAX_UNREDIRECTED_WINDOWS],
            "unredirected_windows_dropped": (
                receipt.get("unredirected_windows_dropped")
                if isinstance(receipt.get("unredirected_windows_dropped"), int)
                else None
            ),
        }
        per_game.append(
            {
                "canonical_row_id": row_id,
                "run_id": candidate["run_id"],
                "game": row.get("game"),
                "harness_arm": row.get("arm"),
                "banked_levels": row.get("levels", 0),
                "banked_progress_event_count": len(row_transitions),
                "redirect_count": len(row_redirects),
                "old_credit_count": sum(int(item["old_credit"]) for item in row_redirects),
                "banked_credit_count": sum(int(item["banked_credit"]) for item in row_redirects),
                "censored_redirect_count": sum(int(item["censored"]) for item in row_redirects),
                "solve_provenance": str(row.get("solve_provenance") or "live_agent_self_discovery"),
            }
        )

    by_transition: dict[str, list[JsonDict]] = defaultdict(list)
    for redirect in redirects:
        if redirect["banked_transition_id"]:
            by_transition[redirect["banked_transition_id"]].append(redirect)
    competing: list[JsonDict] = []
    for transition_id, shared in by_transition.items():
        if len(shared) < 2:
            continue
        for redirect in shared:
            others = [row["redirect_id"] for row in shared if row is not redirect]
            redirect["competing_redirect_ids"] = others
            redirect["competing_redirect_count"] = len(others)
            competing.append(
                {
                    "redirect_id": redirect["redirect_id"],
                    "banked_transition_id": transition_id,
                    "competing_redirect_ids": others,
                    "shared_credit_is_not_causal": True,
                }
            )
    return {
        "redirects": redirects,
        "transitions": transitions,
        "corrected_entries": corrected_entries,
        "per_game": per_game,
        "competing": competing,
    }


def _empty_rows() -> JsonDict:
    return {
        "rows": [],
        "discovered_file_rows": [],
        "provenance_rows": [],
        "dedupe_rows": [],
        "applied_receipt_rows": [],
        "shadow_receipt_rows": [],
        "error_receipt_rows": [],
        "redirect_rows": [],
        "banked_level_transition_rows": [],
        "actions_to_progress_rows": [],
        "censored_rows": [],
        "competing_redirect_rows": [],
        "transient_vs_banked_credit_rows": [],
        "per_game_rows": [],
        "per_arm_rows": [],
        "refinement_recommendation_rows": [],
    }


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    missing_principles = set(artifact) - set(FIELD_PRINCIPLES)
    if missing_principles:
        raise ValueError(f"missing field principles: {sorted(missing_principles)}")
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6921,
        "run_date": run_date,
        "status": "complete_blocked_preconditions",
        "field_principles": {},
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "receipt_discovery_manifest": {"roots": [dict(row) for row in roots], "blocked": True},
        **_empty_rows(),
        "refinement_policy": {"recommendation_only": True, "not_run": "precondition_failure"},
        "refinement_input_receipt_count": 0,
        "automatic_arm_mutation_count": 0,
        "new_eligible_receipt_count": 0,
        "banked_progress_event_count": 0,
        "solve_provenance": "live_agent_self_discovery",
        "random_seed": 6921,
        "reproducibility_checksum": "",
        "arc_supervisor_audit_complete_score": 0,
        "banked_credit_eligible_score": 0,
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_arc_dynamic_supervisor_banked_credit",
        "solve_claim": False,
        "new_solve_count": 0,
        "arc_run_launch_count": 0,
        "game_adapter_mutation_count": 0,
        "registry_mutation_count": 0,
        "source_game_files_read": [],
    }
    return _finish_artifact(artifact)


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    run_date: str,
    duration_s: float,
    discovery_roots: Sequence[Mapping[str, Any]] | None = None,
    expected_prior_hashes: Mapping[str, str] | None = None,
    source_commit_fallback: str | None = None,
) -> JsonDict:
    """Build the terminal audit in memory without changing any evidence file."""

    ledger_payload, _ = _read_json(root / LEDGER_PATH)
    safe_ledger = ledger_payload if isinstance(ledger_payload, dict) else empty_ledger()
    roots = [
        deepcopy(dict(row))
        for row in (discovery_roots or _default_discovery_roots(root, safe_ledger))
    ]
    expected = dict(expected_prior_hashes or EXPECTED_PRIOR_ARTIFACT_HASHES)
    checks, source_hashes, ledger = precondition_checks(root, roots, expected)
    if any(row["passed"] is not True for row in checks):
        return _blocked_artifact(
            run_date=run_date,
            duration_s=duration_s,
            checks=checks,
            hashes=source_hashes,
            roots=roots,
        )

    fallback = source_commit_fallback or _git_head(root)
    discovery = discover_receipts(root, roots, source_commit_fallback=fallback)
    qualified = qualify_and_dedupe(discovery["candidates"])
    selected = qualified["selected"]
    applied = [row for row in selected if row["disposition"] == "applied"]
    shadows = [row for row in selected if row["disposition"] == "shadow"]
    errors = [row for row in selected if row["disposition"] == "error"]
    replay = replay_banked_credit(applied)

    corrected_ledger = empty_ledger()
    corrected_ledger["entries"] = replay["corrected_entries"]
    recommendation = evaluate(corrected_ledger, f"{run_date}T00:00:00+00:00")
    per_arm = [
        {**dict(row), "credit_basis": "first_later_banked_level_transition"}
        for row in recommendation["per_arm"]
    ]
    refinement_rows = [dict(row) for row in recommendation["recommendations"]]
    if recommendation.get("new_arm_specification"):
        refinement_rows.append(
            {"kind": "new_arm_specification", **recommendation["new_arm_specification"]}
        )

    known_ids = set(ledger["entries"])
    new_count = sum(candidate["canonical_row_id"] not in known_ids for candidate in applied)
    banked_event_count = len(replay["transitions"])
    floor_passed = any(row["fired"] > 0 and row["meets_floor"] for row in per_arm)
    banked_eligible = int(new_count > 0 and banked_event_count > 0 and floor_passed)
    audit_checks = [
        _check("receipt_discovery_complete", True, True, True),
        _check("provenance_classification_complete", True, True, True),
        _check("canonical_row_dedupe_complete", True, True, True),
        _check("banked_credit_replay_complete", True, True, True),
        _check("new_eligible_receipt_count", ">0", new_count, new_count > 0),
        _check("banked_progress_event_count", ">0", banked_event_count, banked_event_count > 0),
        _check("frozen_evidence_floor_passed", True, floor_passed, floor_passed),
        _check("automatic_arm_mutation_count", 0, 0, True),
    ]
    if new_count == 0:
        honest_verdict = "complete_no_new_eligible_receipts"
        status = "complete_no_new_eligible_receipts"
    elif not banked_eligible:
        honest_verdict = "complete_insufficient_banked_progress_evidence"
        status = "complete_insufficient_banked_progress_evidence"
    else:
        honest_verdict = "complete_banked_credit_evidence_eligible"
        status = "complete_banked_credit_evidence_eligible"

    source_hashes.update(
        {
            row["path"]: {
                "name": "discovered_receipt_file",
                "path": row["path"],
                "exists": True,
                "file_sha256": row["content_sha256"],
            }
            for row in discovery["files"]
        }
    )
    redirect_rows = replay["redirects"]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6921,
        "run_date": run_date,
        "status": status,
        "field_principles": {},
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_hashes,
        "receipt_discovery_manifest": {
            "roots": discovery["roots"],
            "discovered_receipt_file_count": len(discovery["files"]),
            "discovered_receipt_row_count": len(discovery["candidates"]),
            "canonical_receipt_row_count": len(selected),
        },
        "rows": deepcopy(redirect_rows),
        "discovered_file_rows": discovery["files"],
        "provenance_rows": qualified["provenance_rows"],
        "dedupe_rows": qualified["dedupe_rows"],
        "applied_receipt_rows": [_receipt_projection(row) for row in applied],
        "shadow_receipt_rows": [_receipt_projection(row) for row in shadows],
        "error_receipt_rows": [_receipt_projection(row) for row in errors],
        "redirect_rows": redirect_rows,
        "banked_level_transition_rows": replay["transitions"],
        "actions_to_progress_rows": [
            {
                "redirect_id": row["redirect_id"],
                "old_actions_to_progress": row["old_actions_to_progress"],
                "actions_to_banked_progress": row["actions_to_banked_progress"],
                "mismatch": row["actions_to_progress_mismatch"],
                "authoritative_source": "banked_replay",
            }
            for row in redirect_rows
        ],
        "censored_rows": [dict(row) for row in redirect_rows if row["censored"]],
        "competing_redirect_rows": replay["competing"],
        "transient_vs_banked_credit_rows": [
            {
                "redirect_id": row["redirect_id"],
                "old_resolved_by_levelup": row["old_credit"],
                "old_helped_count": row["old_helped_count"],
                "banked_credit": row["banked_credit"],
                "credit_changed": row["old_credit"] != row["banked_credit"],
            }
            for row in redirect_rows
        ],
        "per_game_rows": replay["per_game"],
        "per_arm_rows": per_arm,
        "refinement_recommendation_rows": refinement_rows,
        "refinement_policy": recommendation,
        "refinement_input_receipt_count": len(applied),
        "automatic_arm_mutation_count": 0,
        "new_eligible_receipt_count": new_count,
        "banked_progress_event_count": banked_event_count,
        "solve_provenance": "live_agent_self_discovery",
        "random_seed": 6921,
        "reproducibility_checksum": "",
        "arc_supervisor_audit_complete_score": 1,
        "banked_credit_eligible_score": banked_eligible,
        "gate_check_summary": _gate_summary(audit_checks),
        "verifier_is_oracle": False,
        "verdict_class": "null",
        "honest_verdict": honest_verdict,
        "solve_claim": False,
        "new_solve_count": 0,
        "arc_run_launch_count": 0,
        "game_adapter_mutation_count": 0,
        "registry_mutation_count": 0,
        "source_game_files_read": [],
    }
    return _finish_artifact(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the task contract without trusting the artifact's verdict."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field: {field}")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles must cover every top-level field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference_substrate")
    if artifact.get("automatic_arm_mutation_count") != 0:
        errors.append("automatic arm mutation is forbidden")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("invalid verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must have the complete_ terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("solve_claim") is not False or artifact.get("new_solve_count") != 0:
        errors.append("the audit must not claim a new solve")
    if (
        artifact.get("arc_run_launch_count") != 0
        or artifact.get("game_adapter_mutation_count") != 0
    ):
        errors.append("the audit must not run ARC or change an adapter")
    if artifact.get("registry_mutation_count") != 0:
        errors.append("the audit must not mutate the registry")
    if any(
        row.get("solve_provenance") != "live_agent_self_discovery"
        for row in artifact.get("per_game_rows", [])
    ):
        errors.append("every game outcome row needs live_agent_self_discovery provenance")
    if artifact.get("refinement_policy", {}).get("recommendation_only") is not True:
        errors.append("refinement policy must remain recommendation-only")
    applied_ids = {row.get("canonical_row_id") for row in artifact.get("applied_receipt_rows", [])}
    excluded_ids = {
        row.get("canonical_row_id")
        for key in ("shadow_receipt_rows", "error_receipt_rows")
        for row in artifact.get(key, [])
    }
    if applied_ids & excluded_ids:
        errors.append("shadow or error row entered applied evidence")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        errors.append("gate_check_summary must be an object")
    elif summary.get("passed") is False and not all(
        key in summary for key in ("failed_check", "expected", "observed")
    ):
        errors.append("failed gate summary lacks exact values")
    return errors


def write_artifact_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace one result atomically so interruption cannot leave partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Execution date in YYYYMMDD form")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scored-root", action="append", default=[])
    parser.add_argument("--lever-root", action="append", default=[])
    args = parser.parse_args(argv)
    explicit_roots = [
        *({"kind": "scored_path", "path": value} for value in args.scored_root),
        *({"kind": "lever_harness", "path": value} for value in args.lever_root),
    ]
    started = time.perf_counter()
    artifact = build_artifact(
        args.root,
        run_date=args.date,
        duration_s=0.0,
        discovery_roots=explicit_roots or None,
    )
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        print("ERROR: " + "; ".join(errors), file=sys.stderr)
        return 1
    output = args.output or (args.root / OUTPUT_PATH)
    write_artifact_atomic(output, artifact)
    print(
        json.dumps(
            {
                "output": _relative_or_absolute(output, args.root),
                "honest_verdict": artifact["honest_verdict"],
                "new_eligible_receipt_count": artifact["new_eligible_receipt_count"],
                "banked_progress_event_count": artifact["banked_progress_event_count"],
                "banked_credit_eligible_score": artifact["banked_credit_eligible_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the thin command wrapper.
    raise SystemExit(main())
