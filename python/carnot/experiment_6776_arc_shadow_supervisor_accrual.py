"""Accrue window-120 shadow-supervisor receipts on the production ARC path.

The experiment treats shadow redirects as counterfactual observations. It
does not relabel them as applied evidence. Each admitted cell owns one worker,
one GPU lease, and one llama.cpp session. A blocked host still emits the full
frozen denominator so a missing live run cannot look like a clean empty run.

Spec refs: REQ-ARC-WMTE-6776 and SCENARIO-ARC-WMTE-6776-1..5.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6764_arc_exclusive_load_preflight as admission
from carnot.agentic import arc_supervisor_refinement as refinement
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.testing import long_run_receipt


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6776_arc_shadow_supervisor_accrual.json"
WORK_DIR = REPO_ROOT / "results/.experiment_6776_arc_shadow_supervisor_accrual"
LEDGER_PATH = REPO_ROOT / "ops/arc_supervisor_refinement_ledger.json"
REGISTRY_PATH = REPO_ROOT / "ops/arc_solve_registry.yaml"
ARCHITECTURE_PATH = REPO_ROOT / "_bmad/architecture.md"
AGENT_PATH = REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py"
SUPERVISOR_PATH = REPO_ROOT / "python/carnot/agentic/arc_trajectory_supervisor.py"
SCHEMA = "carnot.experiment_6776.arc_shadow_supervisor_accrual.v1"
RUN_DATE = "20260830"
RANDOM_SEEDS = (2026083001, 2026083002, 2026083003, 2026083004)
CANARY_SEED = 2026083099
PANEL_GAMES = ("tu93", "ar25")
PUBLIC_GAMES = (
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "dc22",
    "ft09",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "r11l",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "sp80",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "vc33",
    "wa30",
)
ACTION_BUDGET = 399
CANARY_ACTION_BUDGET = 140
ACTION_BLOCK = 20
SUPERVISOR_WINDOW = 120
SUPERVISOR_MODE = "shadow"
CONTEXT_REQUESTED = 32_768
INFERENCE_SUBSTRATE_DETAIL = "production E3AgentPolicy on task-owned local CUDA GGUF"
BLOCKED_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
LIVE_INFERENCE_SUBSTRATE = "live_llm_inference"
EXPECTED_GPU_UUIDS = admission.EXPECTED_GPU_UUIDS
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        **deepcopy(admission.MODEL_SPECS[0]),
        "role": "immutable_scored_arc_generator",
        "decode_tokens": 4096,
    },
    {
        **deepcopy(admission.MODEL_SPECS[1]),
        "role": "flagship_moe_transport_canary",
        "decode_tokens": 512,
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "title",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "inference_substrate_detail",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "models_used",
    "live_model_invoked",
    "frozen_manifest",
    "rows",
    "gpu_receipts",
    "death_receipts",
    "shard_receipts",
    "supervisor_window",
    "supervisor_mode",
    "action_hash_invariance",
    "firings_before_by_arm",
    "firings_after_by_arm",
    "evidence_floor_met_by_arm",
    "supervisor_refinement_receipt",
    "shadow_supervisor_transport_ready",
    "solve_claim",
    "source_receipts",
    "gate_check_summary",
    "preconditions_checked",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned schema lets consumers reject incompatible evidence.",
    "experiment": "The experiment number binds the artifact to REQ-ARC-WMTE-6776.",
    "title": "The title states that this is shadow evidence, not a flag activation.",
    "run_date": "The planning date fixes the frozen run period.",
    "status": "The status separates ready, partial, disqualified, and blocked outcomes.",
    "field_principles": "Each required field states why an auditor needs it.",
    "inference_substrate": "The closed ARC lint enum distinguishes blocked aggregation from live inference.",
    "inference_substrate_detail": "The exact intended production path excludes CPU, remote, legacy, and substituted models.",
    "duration_s": "Monotonic wall time exposes skipped live work.",
    "random_seed": "The complete cell seed schedule makes action choices repeatable.",
    "reproducibility_checksum": "One digest binds models, panel, code, ledger, manifest, and rows.",
    "model_specs": "Exact repository, file, hash, role, and tokenizer pins prevent substitution.",
    "models_used": "Only models that produced a live row appear as used.",
    "live_model_invoked": "First-token evidence distinguishes a live model from a dry receipt.",
    "frozen_manifest": "Games, seeds, budgets, prompts, sampling, and order are fixed before inference.",
    "rows": "One cell-and-arm row keeps the full live or blocked denominator visible.",
    "gpu_receipts": "Lease, offload, VRAM, process, and recovery facts prove owned CUDA work.",
    "death_receipts": "Signal and process attribution makes an interrupted cell diagnosable.",
    "shard_receipts": "Atomic block, firing, and completion checkpoints preserve progress.",
    "supervisor_window": "The value 120 makes a stagnation window reachable within the budget.",
    "supervisor_mode": "Shadow mode records counterfactuals without applying redirects.",
    "action_hash_invariance": "Paired action hashes detect any scored behavior change.",
    "firings_before_by_arm": "The applied ledger baseline prevents shadow inflation.",
    "firings_after_by_arm": "Post-tool applied counts show exactly what ingestion changed.",
    "evidence_floor_met_by_arm": "Per-arm booleans state whether ten applied firings exist.",
    "supervisor_refinement_receipt": "Tool hashes, dedupe, Wilson rows, and recommendation make refinement auditable.",
    "shadow_supervisor_transport_ready": "Exp6777 consumes this exact transport-and-invariance gate.",
    "solve_claim": "False prevents a replay receipt from becoming a new solve claim.",
    "source_receipts": "Hashes and explicit prohibitions show no game source or offline BFS use.",
    "gate_check_summary": "A blocked or failed result names the exact check and observed value.",
    "preconditions_checked": "Host, route, registry, model, helper, and resource observations remain inspectable.",
    "verifier_is_oracle": "False states that transport readiness is not a correctness oracle.",
    "verdict_class": "A closed class prevents an unsupported result from becoming positive.",
    "honest_verdict": "A terminal prefix states readiness and evidence limits without inflation.",
}


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for all content-addressed receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading a multi-gigabyte GGUF into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def named_cached_model_path(path: Path, filename: str) -> str:
    """Return the named Hugging Face snapshot link for a resolved blob receipt."""

    if path.name == filename:
        return str(path)
    model_root = path.parent.parent if path.parent.name == "blobs" else None
    if model_root is not None:
        for candidate in sorted(model_root.glob(f"snapshots/*/{filename}")):
            if candidate.resolve() == path.resolve():
                return str(candidate)
    return str(path)


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every artifact field except the self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def _default_cells() -> list[JsonDict]:
    """Return the fixed live order, including one invariance pair and one MoE canary."""

    cells: list[JsonDict] = []
    pair_id = f"{PANEL_GAMES[0]}:{RANDOM_SEEDS[0]}"
    for game in PANEL_GAMES:
        for seed in RANDOM_SEEDS:
            cells.append(
                {
                    "cell_id": f"science:{game}:{seed}:shadow",
                    "cell_kind": "science",
                    "game": game,
                    "seed": seed,
                    "model_id": MODEL_SPECS[0]["model_id"],
                    "model_role": MODEL_SPECS[0]["role"],
                    "action_budget": ACTION_BUDGET,
                    "supervisor_observation": "shadow",
                    "invariance_pair_id": pair_id
                    if game == PANEL_GAMES[0] and seed == RANDOM_SEEDS[0]
                    else None,
                }
            )
    cells.append(
        {
            "cell_id": f"canary:{PANEL_GAMES[0]}:{RANDOM_SEEDS[0]}:shadow_off",
            "cell_kind": "shadow_off_canary",
            "game": PANEL_GAMES[0],
            "seed": RANDOM_SEEDS[0],
            "model_id": MODEL_SPECS[0]["model_id"],
            "model_role": MODEL_SPECS[0]["role"],
            "action_budget": ACTION_BUDGET,
            "supervisor_observation": "off",
            "invariance_pair_id": pair_id,
        }
    )
    cells.append(
        {
            "cell_id": f"transport:{PANEL_GAMES[1]}:{CANARY_SEED}:shadow",
            "cell_kind": "transport_canary",
            "game": PANEL_GAMES[1],
            "seed": CANARY_SEED,
            "model_id": MODEL_SPECS[1]["model_id"],
            "model_role": MODEL_SPECS[1]["role"],
            "action_budget": CANARY_ACTION_BUDGET,
            "supervisor_observation": "shadow",
            "invariance_pair_id": None,
        }
    )
    return cells


def freeze_manifest(*, cells: Sequence[Mapping[str, Any]] | None = None) -> JsonDict:
    """Freeze the panel and every setting that can change generated actions."""

    manifest: JsonDict = {
        "schema": SCHEMA + ".frozen_manifest",
        "requirement": "REQ-ARC-WMTE-6776",
        "games": list(PANEL_GAMES),
        "public_registry_games_checked": list(PUBLIC_GAMES),
        "seeds": list(RANDOM_SEEDS),
        "canary_seed": CANARY_SEED,
        "action_budget": ACTION_BUDGET,
        "canary_action_budget": CANARY_ACTION_BUDGET,
        "action_block": ACTION_BLOCK,
        "supervisor_window": SUPERVISOR_WINDOW,
        "supervisor_mode": SUPERVISOR_MODE,
        "context_requested": CONTEXT_REQUESTED,
        "prompt_contract": "production E3AgentPolicy prompt builders; no task-authored game prompt",
        "sampling": {
            "generator_seed_per_cell": True,
            "mtp": False,
            "kv_quant": "q8_0",
            "no_think_prefix": "/no_think\\n",
            "decode_tokens_by_role": {spec["role"]: spec["decode_tokens"] for spec in MODEL_SPECS},
        },
        "cell_order": "listed order; one fresh worker, lease, port, and model session per cell",
        "cells": [deepcopy(dict(row)) for row in (cells or _default_cells())],
        "quality_reducer_model_role": MODEL_SPECS[0]["role"],
        "excluded_quality_model_role": MODEL_SPECS[1]["role"],
        "solve_target": None,
        "solve_claim": False,
        "game_source_access": False,
        "game_adapter_added": False,
        "offline_bfs_used": False,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def scored_action_hash(actions: Sequence[Mapping[str, Any]]) -> str:
    """Hash scored actions only, excluding shadow diagnostics and timings."""

    return sha256_json([deepcopy(dict(action)) for action in actions])


def action_hash_invariance(
    cells: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> JsonDict:
    """Compare each frozen shadow cell with its supervisor-off replay canary."""

    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        pair = cell.get("invariance_pair_id")
        if pair:
            by_pair.setdefault(str(pair), []).append(cell)
    pairs = []
    for pair_id in sorted(by_pair):
        candidates = by_pair[pair_id]
        shadow = next(
            (row for row in candidates if row.get("supervisor_observation") == "shadow"), None
        )
        control = next(
            (row for row in candidates if row.get("supervisor_observation") == "off"), None
        )
        shadow_hash = shadow.get("scored_action_hash") if shadow else None
        control_hash = control.get("scored_action_hash") if control else None
        passed = bool(
            shadow
            and control
            and shadow.get("status") == "complete"
            and control.get("status") == "complete"
            and shadow_hash
            and shadow_hash == control_hash
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "shadow_cell_id": shadow.get("cell_id") if shadow else None,
                "control_cell_id": control.get("cell_id") if control else None,
                "shadow_action_hash": shadow_hash,
                "control_action_hash": control_hash,
                "passed": passed,
            }
        )
    expected_pairs = {
        str(row["invariance_pair_id"])
        for row in manifest.get("cells", [])
        if row.get("invariance_pair_id")
    }
    return {
        "method": "same game, seed, model, budget, prompt, context, and sampling; shadow versus observation disabled",
        "expected_pair_ids": sorted(expected_pairs),
        "pairs": pairs,
        "passed": bool(expected_pairs)
        and {row["pair_id"] for row in pairs} == expected_pairs
        and all(row["passed"] is True for row in pairs),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    """Publish complete JSON through a same-directory atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_progress_shard(
    path: Path,
    *,
    cell_id: str,
    action_index: int,
    event: str,
    trajectory_supervisor: Mapping[str, Any],
) -> JsonDict:
    """Atomically retain cumulative block and firing progress for one cell."""

    prior_events: list[JsonDict] = []
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
            prior_events = list(loaded.get("events") or []) if isinstance(loaded, dict) else []
        except (OSError, ValueError, json.JSONDecodeError):
            prior_events = []
    row = {
        "event": str(event),
        "action_index": int(action_index),
        "trajectory_supervisor_sha256": sha256_json(trajectory_supervisor),
    }
    payload = {
        "schema": SCHEMA + ".progress_shard",
        "cell_id": str(cell_id),
        **row,
        "events": [*prior_events, row],
    }
    write_json_atomic(path, payload)
    return {
        "path": str(path),
        "event": str(event),
        "action_index": int(action_index),
        "atomic_replace": True,
        "sha256": sha256_file(path),
    }


def install_death_receipt(path: Path, progress: Callable[[], object] | None = None) -> JsonDict:
    """Install the shared signal receipt and state its exact destination."""

    long_run_receipt.install(path, progress=progress)
    return {"installed": True, "path": str(path), "signal_receipt": None}


def model_pin_errors(models: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject missing, reordered, substituted, or tokenizer-less model rows."""

    errors: list[str] = []
    by_id = {str(row.get("model_id")): row for row in models}
    if list(by_id) != [spec["model_id"] for spec in MODEL_SPECS]:
        errors.append("model_order_or_denominator")
    for spec in MODEL_SPECS:
        row = by_id.get(spec["model_id"])
        if row is None:
            errors.append(f"model_missing:{spec['model_id']}")
            continue
        if row.get("role") != spec["role"]:
            errors.append(f"model_role:{spec['model_id']}")
        if Path(str(row.get("model_path") or "")).name != spec["filename"]:
            errors.append(f"model_filename:{spec['model_id']}")
        if row.get("model_sha256") != spec["expected_sha256"]:
            errors.append(f"model_sha256:{spec['model_id']}")
        if row.get("resolved") is not True:
            errors.append(f"model_resolution:{spec['model_id']}")
        tokenizer = row.get("tokenizer") if isinstance(row.get("tokenizer"), Mapping) else {}
        if (
            tokenizer.get("source") != "llama.cpp_embedded_gguf"
            or tokenizer.get("loadable") is not True
        ):
            errors.append(f"embedded_tokenizer:{spec['model_id']}")
    return errors


def teardown_errors(cell: Mapping[str, Any]) -> list[str]:
    """Name any owned process, lease, port, or VRAM resource left behind."""

    errors: list[str] = []
    worker = cell.get("worker_process") if isinstance(cell.get("worker_process"), Mapping) else {}
    server = (
        cell.get("llama_server_process")
        if isinstance(cell.get("llama_server_process"), Mapping)
        else {}
    )
    gpu = cell.get("gpu_receipt") if isinstance(cell.get("gpu_receipt"), Mapping) else {}
    if worker.get("exit_code") != 0 or worker.get("absent_after_exit") is not True:
        errors.append("worker_process")
    if server.get("absent_after_exit") is not True:
        errors.append("llama_server_process")
    if (gpu.get("lease_release") or {}).get("released") is not True:
        errors.append("lease_release")
    if (gpu.get("vram_recovery") or {}).get("passed") is not True:
        errors.append("vram_recovery")
    if (gpu.get("port_release") or {}).get("closed") is not True:
        errors.append("port_release")
    if gpu.get("unrelated_processes_signaled") not in ([], ()):
        errors.append("unrelated_process_signal")
    return errors


def _outcome_for_arm(cell: Mapping[str, Any], arm: str) -> JsonDict:
    receipt = cell.get("trajectory_supervisor")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    outcomes = receipt.get("would_have_arm_outcomes")
    outcomes = outcomes if isinstance(outcomes, Mapping) else {}
    outcome = outcomes.get(arm)
    outcome = outcome if isinstance(outcome, Mapping) else {}
    redirects = receipt.get("would_have_redirects")
    redirects = redirects if isinstance(redirects, list) else []
    return {
        "arm_fired": int(outcome.get("fired") or 0),
        "arm_helped_counterfactual": int(outcome.get("helped") or 0),
        "arm_redirect_receipts": [
            deepcopy(dict(row))
            for row in redirects
            if isinstance(row, Mapping) and row.get("arm") == arm
        ],
    }


def expand_cell_rows(cells: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Emit one compact artifact row for every cell and supervisor arm."""

    rows: list[JsonDict] = []
    for cell in cells:
        for arm in ARM_ORDER:
            row: JsonDict = {
                "row_id": f"{cell.get('cell_id')}:{arm}",
                "cell_id": cell.get("cell_id"),
                "cell_kind": cell.get("cell_kind"),
                "game": cell.get("game"),
                "seed": cell.get("seed"),
                "model_id": cell.get("model_id"),
                "model_role": cell.get("model_role"),
                "supervisor_arm": arm,
                **_outcome_for_arm(cell, arm),
                "supervisor_mode_observed": (cell.get("trajectory_supervisor") or {}).get("mode"),
                "supervisor_window_observed": (cell.get("trajectory_supervisor") or {}).get(
                    "window"
                ),
                "actions_observed": (cell.get("trajectory_supervisor") or {}).get(
                    "actions_observed"
                ),
                "scored_action_hash": cell.get("scored_action_hash"),
                "live_model_invoked": cell.get("live_model_invoked") is True,
                "first_token_observed": cell.get("first_token_observed") is True,
                "worker_pid": (cell.get("worker_process") or {}).get("pid"),
                "llama_server_pid": (cell.get("llama_server_process") or {}).get("pid"),
                "teardown_passed": cell.get("teardown_passed") is True,
                "shard_count": len(cell.get("shard_receipts") or []),
                "failure_class": cell.get("failure_class"),
                "solve_claim": False,
                "cell_receipt_sha256": sha256_json(cell),
                "row_sha256": "",
            }
            row["row_sha256"] = sha256_json(
                {key: value for key, value in row.items() if key != "row_sha256"}
            )
            rows.append(row)
    return rows


def refine_shards(*, ledger_path: Path, inputs: Sequence[Path], now_iso: str) -> JsonDict:
    """Run the existing reducer logic while retaining exact pre/post ledger facts."""

    ledger = refinement.load_ledger(ledger_path)
    before_hash = sha256_file(ledger_path) if ledger_path.is_file() else sha256_json(ledger)
    before_ids = sorted(ledger["entries"])
    before_recommendation = refinement.evaluate(ledger, now_iso)
    files = refinement.scan_inputs(inputs)
    counts = refinement.ingest_files(ledger, files, now_iso)
    recommendation = refinement.evaluate(ledger, now_iso)
    recommendation["ingest_counts"] = counts
    ledger["recommendation"] = recommendation
    if ledger.get("created_at") is None:
        ledger["created_at"] = now_iso
    ledger["updated_at"] = now_iso
    refinement.save_ledger(ledger, ledger_path)
    after_ids = sorted(ledger["entries"])
    return {
        "tool": "scripts/arc_supervisor_refine.py",
        "ran": True,
        "exit_code": 0,
        "inputs": [str(path) for path in files],
        "ledger_path": str(ledger_path),
        "ledger_sha256_before": before_hash,
        "ledger_sha256_after": sha256_file(ledger_path),
        "entry_ids_before": before_ids,
        "entry_ids_after": after_ids,
        "deduplicated": len(after_ids) == len(before_ids) + counts["applied_new"],
        "ingest_counts": counts,
        "firings_before_by_arm": {
            row["arm"]: row["fired"] for row in before_recommendation["per_arm"]
        },
        "recommendation": recommendation,
    }


def _recommendation_counts(receipt: Mapping[str, Any]) -> tuple[JsonDict, JsonDict, JsonDict]:
    recommendation = receipt.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, Mapping) else {}
    rows = recommendation.get("per_arm")
    rows = rows if isinstance(rows, list) else []
    after = {arm: 0 for arm in ARM_ORDER}
    floors = {arm: False for arm in ARM_ORDER}
    for row in rows:
        if isinstance(row, Mapping) and row.get("arm") in after:
            arm = str(row["arm"])
            after[arm] = int(row.get("fired") or 0)
            floors[arm] = row.get("meets_floor") is True
    before_raw = receipt.get("firings_before_by_arm")
    before = (
        {arm: int(before_raw.get(arm, 0)) for arm in ARM_ORDER}
        if isinstance(before_raw, Mapping)
        else deepcopy(after)
    )
    return before, after, floors


def _blocked_cells(
    manifest: Mapping[str, Any], models: Sequence[Mapping[str, Any]], failed_check: str
) -> list[JsonDict]:
    """Keep every frozen cell when admission fails before worker launch."""

    by_id = {str(row.get("model_id")): row for row in models}
    cells: list[JsonDict] = []
    for spec in manifest.get("cells", []):
        model = by_id.get(str(spec.get("model_id")), {})
        shadow = spec.get("supervisor_observation") == "shadow"
        cells.append(
            {
                **deepcopy(dict(spec)),
                "status": "blocked",
                "model_path": model.get("model_path"),
                "model_sha256": model.get("model_sha256"),
                "worker_process": {"pid": None, "exit_code": None, "absent_after_exit": True},
                "llama_server_process": {"pid": None, "exit_code": None, "absent_after_exit": True},
                "llama_server_log": None,
                "live_model_invoked": False,
                "first_token_observed": False,
                "context_observed": None,
                "trajectory_supervisor": {
                    "enabled": False,
                    "mode": "shadow" if shadow else "off",
                    "window": SUPERVISOR_WINDOW if shadow else None,
                    "actions_observed": 0,
                    "would_have_redirects": [] if shadow else None,
                    "would_have_arm_outcomes": (
                        {arm: {"fired": 0, "helped": 0} for arm in ARM_ORDER} if shadow else None
                    ),
                },
                "scored_action_hash": None,
                "scored_actions": 0,
                "levels": 0,
                "gpu_receipt": None,
                "death_receipt": {"installed": False, "path": None, "signal_receipt": None},
                "shard_receipts": [],
                "teardown_passed": False,
                "failure_class": f"preflight_blocked:{failed_check}",
                "solve_claim": False,
            }
        )
    return cells


def _not_run_refinement_receipt(now_iso: str) -> JsonDict:
    """Read the baseline without mutating it when preconditions block workers."""

    ledger = refinement.load_ledger(LEDGER_PATH)
    recommendation = refinement.evaluate(ledger, now_iso)
    ledger_hash = sha256_file(LEDGER_PATH) if LEDGER_PATH.is_file() else sha256_json(ledger)
    entry_ids = sorted(ledger["entries"])
    return {
        "tool": "scripts/arc_supervisor_refine.py",
        "ran": False,
        "reason": "precondition_failed",
        "exit_code": None,
        "inputs": [],
        "ledger_path": str(LEDGER_PATH),
        "ledger_sha256_before": ledger_hash,
        "ledger_sha256_after": ledger_hash,
        "entry_ids_before": entry_ids,
        "entry_ids_after": entry_ids,
        "deduplicated": True,
        "ingest_counts": {
            "files_read": 0,
            "rows_seen": 0,
            "applied_new": 0,
            "applied_duplicate": 0,
            "shadow_observed": 0,
        },
        "firings_before_by_arm": {row["arm"]: row["fired"] for row in recommendation["per_arm"]},
        "recommendation": recommendation,
    }


def _transport_errors(
    *,
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    refinement_receipt: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    if preflight.get("all_passed") is not True:
        errors.append("preconditions")
    errors.extend(model_pin_errors(models))
    expected_ids = [row.get("cell_id") for row in manifest.get("cells", [])]
    if [row.get("cell_id") for row in cells] != expected_ids:
        errors.append("cell_denominator_or_order")
    for cell in cells:
        if cell.get("status") != "complete":
            errors.append(f"cell_status:{cell.get('cell_id')}")
            continue
        if (
            cell.get("live_model_invoked") is not True
            or cell.get("first_token_observed") is not True
        ):
            errors.append(f"first_token:{cell.get('cell_id')}")
        if int(cell.get("context_observed") or 0) < CONTEXT_REQUESTED:
            errors.append(f"context:{cell.get('cell_id')}")
        if cell.get("solve_claim") is not False:
            errors.append(f"solve_claim:{cell.get('cell_id')}")
        if teardown_errors(cell):
            errors.append(f"teardown:{cell.get('cell_id')}")
        gpu = cell.get("gpu_receipt") if isinstance(cell.get("gpu_receipt"), Mapping) else {}
        layers = gpu.get("gpu_layers") if isinstance(gpu.get("gpu_layers"), Mapping) else {}
        if int(layers.get("offloaded") or 0) <= 0:
            errors.append(f"cuda_offload:{cell.get('cell_id')}")
        death = cell.get("death_receipt") if isinstance(cell.get("death_receipt"), Mapping) else {}
        if death.get("installed") is not True:
            errors.append(f"death_receipt:{cell.get('cell_id')}")
        shards = cell.get("shard_receipts") if isinstance(cell.get("shard_receipts"), list) else []
        if not shards or not all(row.get("atomic_replace") is True for row in shards):
            errors.append(f"shards:{cell.get('cell_id')}")
        if cell.get("supervisor_observation") == "shadow":
            receipt = cell.get("trajectory_supervisor")
            receipt = receipt if isinstance(receipt, Mapping) else {}
            if receipt.get("mode") != "shadow" or receipt.get("window") != SUPERVISOR_WINDOW:
                errors.append(f"shadow_receipt:{cell.get('cell_id')}")
            if "redirects" in receipt or "arm_outcomes" in receipt:
                errors.append(f"applied_keys_in_shadow:{cell.get('cell_id')}")
    invariance = action_hash_invariance(cells, manifest)
    if invariance.get("passed") is not True:
        errors.append("action_hash_invariance")
    if refinement_receipt.get("ran") is not True or refinement_receipt.get("exit_code") != 0:
        errors.append("refinement_not_run")
    if refinement_receipt.get("deduplicated") is not True:
        errors.append("ledger_deduplication")
    shadow_cells = sum(1 for row in cells if row.get("supervisor_observation") == "shadow")
    counts = refinement_receipt.get("ingest_counts")
    counts = counts if isinstance(counts, Mapping) else {}
    if int(counts.get("shadow_observed") or 0) < shadow_cells:
        errors.append("shadow_ingestion")
    if int(counts.get("applied_new") or 0) != 0:
        errors.append("shadow_relabelled_applied")
    return errors


def failed_preflight_check(preflight: Mapping[str, Any]) -> Mapping[str, Any]:
    """Choose the most informative failed gate, retaining its observed host inventory."""

    failed = [row for row in preflight.get("checks", []) if row.get("passed") is not True]
    return next(
        (row for row in failed if row.get("check") == "exclusive_gpu_without_unrelated_compute"),
        failed[0]
        if failed
        else {"check": "unknown_precondition", "expected": True, "observed": None},
    )


def build_artifact(
    *,
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    refinement_receipt: Mapping[str, Any],
    duration_s: float,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Reduce retained cells into one terminal, content-addressed artifact."""

    blocked = preflight.get("all_passed") is not True
    invariance = (
        {
            "method": "not run because preconditions failed",
            "expected_pair_ids": [],
            "pairs": [],
            "passed": False,
        }
        if blocked
        else action_hash_invariance(cells, manifest)
    )
    transport_errors = _transport_errors(
        manifest=manifest,
        models=models,
        cells=cells,
        preflight=preflight,
        refinement_receipt=refinement_receipt,
    )
    before, after, floors = _recommendation_counts(refinement_receipt)
    ready = not transport_errors
    if blocked:
        failed = failed_preflight_check(preflight)
        status = "complete_blocked_shadow_supervisor_accrual"
        verdict_class = "blocked"
        honest_verdict = f"complete_blocked_shadow_supervisor_accrual:{failed.get('check')}"
        gate = {
            "passed": False,
            "failed_check": failed.get("check"),
            "expected": deepcopy(failed.get("expected")),
            "observed": deepcopy(failed.get("observed")),
        }
    elif invariance.get("passed") is not True:
        status = "complete_disqualified_shadow_supervisor_action_change"
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_shadow_supervisor_action_change"
        gate = {
            "passed": False,
            "failed_check": "action_hash_invariance",
            "expected": True,
            "observed": invariance,
        }
    elif not ready:
        status = "complete_partial_shadow_supervisor_accrual"
        verdict_class = "partial"
        honest_verdict = f"complete_partial_shadow_supervisor_accrual:{transport_errors[0]}"
        gate = {
            "passed": False,
            "failed_check": transport_errors[0],
            "expected": "live shadow receipts, invariance, ingestion, checkpoints, and teardown",
            "observed": transport_errors,
        }
    elif not all(floors.values()):
        status = "complete_shadow_supervisor_transport_ready_evidence_floor_partial"
        verdict_class = "partial"
        honest_verdict = "complete_shadow_supervisor_transport_ready_evidence_floor_partial"
        gate = {
            "passed": True,
            "failed_check": None,
            "expected": "transport and invariance ready; evidence floor reported independently",
            "observed": {"transport_ready": True, "evidence_floor_met_by_arm": floors},
        }
    else:
        status = "complete_shadow_supervisor_transport_and_evidence_ready"
        verdict_class = "null"
        honest_verdict = "complete_shadow_supervisor_transport_and_evidence_ready"
        gate = {"passed": True, "failed_check": None, "expected": True, "observed": True}

    expanded = expand_cell_rows(cells)
    used_ids = {
        str(cell.get("model_id")) for cell in cells if cell.get("live_model_invoked") is True
    }
    source_receipts = deepcopy(dict(preflight.get("source_receipts") or {}))
    source_receipts.update(
        {
            "agent_source": {"path": str(AGENT_PATH), "sha256": sha256_file(AGENT_PATH)},
            "supervisor_source": {
                "path": str(SUPERVISOR_PATH),
                "sha256": sha256_file(SUPERVISOR_PATH),
            },
            "refinement_ledger": {
                "path": str(LEDGER_PATH),
                "sha256_before": refinement_receipt.get("ledger_sha256_before"),
                "sha256_after": refinement_receipt.get("ledger_sha256_after"),
            },
            "game_source_accessed": False,
            "offline_bfs_used": False,
            "game_adapter_added": False,
            "solve_registry_edited": False,
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6776,
        "title": "Window-120 shadow-supervisor evidence accrual",
        "run_date": str(run_date),
        "status": status,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": (
            LIVE_INFERENCE_SUBSTRATE if used_ids else BLOCKED_INFERENCE_SUBSTRATE
        ),
        "inference_substrate_detail": INFERENCE_SUBSTRATE_DETAIL,
        "duration_s": round(max(0.0, float(duration_s)), 6),
        "random_seed": {
            "cell_seed_schedule": [row.get("seed") for row in manifest.get("cells", [])],
            "frozen_science_seeds": list(manifest.get("seeds") or []),
            "canary_seed": manifest.get("canary_seed"),
        },
        "reproducibility_checksum": "",
        "model_specs": [deepcopy(dict(row)) for row in models],
        "models_used": [
            deepcopy(dict(row)) for row in models if str(row.get("model_id")) in used_ids
        ],
        "live_model_invoked": bool(used_ids),
        "frozen_manifest": deepcopy(dict(manifest)),
        "rows": expanded,
        "gpu_receipts": [
            {"cell_id": row.get("cell_id"), "receipt": deepcopy(row.get("gpu_receipt"))}
            for row in cells
        ],
        "death_receipts": [
            {"cell_id": row.get("cell_id"), "receipt": deepcopy(row.get("death_receipt"))}
            for row in cells
        ],
        "shard_receipts": [
            {"cell_id": row.get("cell_id"), "receipts": deepcopy(row.get("shard_receipts") or [])}
            for row in cells
        ],
        "supervisor_window": SUPERVISOR_WINDOW,
        "supervisor_mode": SUPERVISOR_MODE,
        "action_hash_invariance": invariance,
        "firings_before_by_arm": before,
        "firings_after_by_arm": after,
        "evidence_floor_met_by_arm": floors,
        "supervisor_refinement_receipt": deepcopy(dict(refinement_receipt)),
        "shadow_supervisor_transport_ready": ready,
        "solve_claim": False,
        "source_receipts": source_receipts,
        "gate_check_summary": gate,
        "preconditions_checked": deepcopy(dict(preflight)),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return every schema, reduction, provenance, or claim error."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_fields")
    if set(artifact) - set(artifact.get("field_principles", {})):
        errors.append("field_principles")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    expected_substrate = (
        LIVE_INFERENCE_SUBSTRATE
        if artifact.get("live_model_invoked") is True
        else BLOCKED_INFERENCE_SUBSTRATE
    )
    if artifact.get("inference_substrate") != expected_substrate:
        errors.append("inference_substrate")
    if artifact.get("inference_substrate_detail") != INFERENCE_SUBSTRATE_DETAIL:
        errors.append("inference_substrate_detail")
    if artifact.get("supervisor_window") != SUPERVISOR_WINDOW:
        errors.append("supervisor_window")
    if artifact.get("supervisor_mode") != SUPERVISOR_MODE:
        errors.append("supervisor_mode")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if not re.match(
        r"^(complete:|complete_|success:|success_|passed:|passed_|shipped:|shipped_)",
        str(artifact.get("honest_verdict") or ""),
    ):
        errors.append("honest_verdict")
    manifest = artifact.get("frozen_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    expected_manifest_hash = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        errors.append("manifest_sha256")
    model_specs = artifact.get("model_specs")
    model_specs = model_specs if isinstance(model_specs, list) else []
    errors.extend(model_pin_errors(model_specs))
    rows = artifact.get("rows") if isinstance(artifact.get("rows"), list) else []
    expected_ids = [
        f"{cell.get('cell_id')}:{arm}" for cell in manifest.get("cells", []) for arm in ARM_ORDER
    ]
    if [row.get("row_id") for row in rows] != expected_ids:
        errors.append("row_denominator_or_order")
    for row in rows:
        if row.get("solve_claim") is not False:
            errors.append(f"row_solve_claim:{row.get('row_id')}")
        expected_hash = sha256_json(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        if row.get("row_sha256") != expected_hash:
            errors.append(f"row_sha256:{row.get('row_id')}")
    receipt = artifact.get("supervisor_refinement_receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    before, after, floors = _recommendation_counts(receipt)
    if artifact.get("firings_before_by_arm") != before:
        errors.append("firings_before_by_arm")
    if artifact.get("firings_after_by_arm") != after:
        errors.append("firings_after_by_arm")
    if artifact.get("evidence_floor_met_by_arm") != floors:
        errors.append("evidence_floor_met_by_arm")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        if artifact.get("shadow_supervisor_transport_ready") is not False:
            errors.append("blocked_transport_ready")
        if not all(
            str(row.get("failure_class") or "").startswith("preflight_blocked:") for row in rows
        ):
            errors.append("blocked_rows")
    invariance = artifact.get("action_hash_invariance")
    invariance = invariance if isinstance(invariance, Mapping) else {}
    expected_pairs = sorted(
        {
            str(row["invariance_pair_id"])
            for row in manifest.get("cells", [])
            if row.get("invariance_pair_id")
        }
    )
    pair_rows = invariance.get("pairs") if isinstance(invariance.get("pairs"), list) else []
    structurally_invariant = (
        bool(expected_pairs)
        and invariance.get("expected_pair_ids") == expected_pairs
        and all(
            isinstance(row, Mapping)
            and row.get("pair_id") in expected_pairs
            and row.get("shadow_action_hash")
            and row.get("shadow_action_hash") == row.get("control_action_hash")
            and row.get("passed") is True
            for row in pair_rows
        )
    )
    if not blocked and invariance.get("passed") != structurally_invariant:
        errors.append("action_hash_invariance")
    if artifact.get("shadow_supervisor_transport_ready") is True and not structurally_invariant:
        errors.append("transport_ready_without_invariance")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def collect_preconditions(
    root: Path = REPO_ROOT, *, run_date: str = RUN_DATE
) -> JsonDict:  # pragma: no cover - live host boundary
    """Extend the existing exclusive-load gate with supervisor and registry checks."""

    base = admission.collect_preconditions(root)
    models = []
    by_id = {str(row.get("model_id")): row for row in base.get("models", [])}
    for spec in MODEL_SPECS:
        model = {**deepcopy(by_id.get(spec["model_id"], {})), **deepcopy(spec)}
        model["model_path"] = named_cached_model_path(
            Path(str(model.get("model_path") or "")), str(spec["filename"])
        )
        models.append(model)
    checks = list(base.get("checks", []))
    try:
        from carnot.agentic.arc_competition_agent import (
            E3AgentPolicy,
            _make_trajectory_supervisor,
            make_carnot_agent,
        )

        old_flag = os.environ.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR", None)
        old_window = os.environ.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", None)
        supervisor, applies = _make_trajectory_supervisor()
        if old_flag is not None:
            os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR"] = old_flag
        if old_window is not None:
            os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW"] = old_window
        route = callable(make_carnot_agent) and isinstance(E3AgentPolicy, type)
        default_receipt = {"window": supervisor.window, "applies": applies}
    except Exception as exc:  # noqa: BLE001
        route = False
        default_receipt = {"error": f"{type(exc).__name__}: {exc}"}
    checks.extend(
        [
            {
                "check": "production_e3_policy_path",
                "expected": True,
                "observed": route,
                "passed": route,
            },
            {
                "check": "supervisor_default_window_and_shadow",
                "expected": {"window": 120, "applies": False},
                "observed": default_receipt,
                "passed": default_receipt == {"window": 120, "applies": False},
            },
            {
                "check": "supervisor_applied_flag_unset",
                "expected": None,
                "observed": os.environ.get("CARNOT_ARC_TRAJECTORY_SUPERVISOR"),
                "passed": os.environ.get("CARNOT_ARC_TRAJECTORY_SUPERVISOR") is None,
            },
            {
                "check": "death_receipt_helper",
                "expected": True,
                "observed": callable(long_run_receipt.install),
                "passed": callable(long_run_receipt.install),
            },
        ]
    )
    architecture_text = ARCHITECTURE_PATH.read_text()
    match = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", architecture_text)
    reconciled = date.fromisoformat(match.group(1)) if match else None
    planning = date.fromisoformat(f"{run_date[:4]}-{run_date[4:6]}-{run_date[6:]}")
    age = (planning - reconciled).days if reconciled else None
    checks.append(
        {
            "check": "architecture_map_fresh",
            "expected": "0 to 30 days old",
            "observed": match.group(1) if match else None,
            "age_days": age,
            "passed": age is not None and 0 <= age <= 30,
        }
    )
    import yaml

    registry_doc = yaml.safe_load(REGISTRY_PATH.read_text())
    registry_games = {
        str(row.get("game")) for row in registry_doc if isinstance(row, Mapping) and row.get("game")
    }
    registry_ok = set(PUBLIC_GAMES) <= registry_games and freeze_manifest()["solve_target"] is None
    checks.append(
        {
            "check": "registry_precheck_all_public_games_no_new_target",
            "expected": {"public_games": list(PUBLIC_GAMES), "solve_target": None},
            "observed": {
                "games_present": sorted(set(PUBLIC_GAMES) & registry_games),
                "solve_target": None,
            },
            "passed": registry_ok,
        }
    )
    eligible_exclusive = [
        row
        for row in base.get("device_inventory_before", [])
        if int(row.get("memory_free_mb", 0)) >= admission.FROZEN_FREE_VRAM_THRESHOLD_MB
        and not row.get("active_compute_processes")
    ]
    checks.append(
        {
            "check": "exclusive_gpu_without_unrelated_compute",
            "expected": "one RTX 3090 with no active compute owner and at least 22610 MiB free",
            "observed": base.get("device_inventory_before", []),
            "passed": bool(eligible_exclusive),
        }
    )
    ledger = refinement.load_ledger(LEDGER_PATH)
    checks.append(
        {
            "check": "current_refinement_ledger",
            "expected": refinement.LEDGER_SCHEMA,
            "observed": ledger.get("schema"),
            "entries": len(ledger.get("entries", {})),
            "passed": ledger.get("schema") == refinement.LEDGER_SCHEMA,
        }
    )
    base["models"] = models
    base["checks"] = checks
    base["all_passed"] = all(row.get("passed") is True for row in checks)
    base.setdefault("source_receipts", {}).update(
        {
            "architecture_map": {
                "path": str(ARCHITECTURE_PATH),
                "sha256": sha256_file(ARCHITECTURE_PATH),
            },
            "solve_registry": {"path": str(REGISTRY_PATH), "sha256": sha256_file(REGISTRY_PATH)},
            "refinement_ledger": {"path": str(LEDGER_PATH), "sha256": sha256_file(LEDGER_PATH)},
        }
    )
    return base


def run_cell_subprocess(
    cell: Mapping[str, Any], model: Mapping[str, Any], device: Mapping[str, Any], work_root: Path
) -> JsonDict:  # pragma: no cover - live worker boundary
    """Launch one cell in its own process and retain logs and teardown identity."""

    cell_root = work_root / str(cell["cell_id"]).replace(":", "_")
    cell_root.mkdir(parents=True, exist_ok=True)
    job_path = cell_root / "job.json"
    output_path = cell_root / "cell.json"
    stdout_path = cell_root / "worker.stdout.log"
    stderr_path = cell_root / "worker.stderr.log"
    port = admission.choose_free_ports(1)[0]
    write_json_atomic(
        job_path,
        {
            "cell": dict(cell),
            "model": dict(model),
            "device": dict(device),
            "port": port,
            "cell_root": str(cell_root),
        },
    )
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6776_arc_shadow_supervisor_accrual",
        "--worker-job",
        str(job_path),
        "--worker-output",
        str(output_path),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "python") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        try:
            exit_code = process.wait(timeout=max(1800, int(cell["action_budget"]) * 30))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=10)
    if output_path.is_file():
        result = json.loads(output_path.read_text())
    else:
        result = {
            **dict(cell),
            "status": "failed",
            "failure_class": f"worker_exit:{exit_code}",
            "solve_claim": False,
        }
    result["worker_process"] = {
        "pid": process.pid,
        "exit_code": exit_code,
        "absent_after_exit": process.poll() is not None,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    result["teardown_passed"] = not teardown_errors(result)
    write_json_atomic(output_path, result)
    return result


def _worker_entry(
    job_path: Path, output_path: Path
) -> int:  # pragma: no cover - live CUDA boundary
    """Run one leased production E3 cell and stop only its owned server."""

    import random
    import socket

    import numpy as np
    import arc_leaderboard_eval as leaderboard

    from carnot import gpu_lease_phase_journal as lease_api
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    job = json.loads(job_path.read_text())
    cell, model, device = job["cell"], job["model"], job["device"]
    port, cell_root = int(job["port"]), Path(job["cell_root"])
    shard_path = cell_root / "rows.json"
    death_path = cell_root / "death.json"
    progress = {"cell_id": cell["cell_id"], "actions": 0, "firings": 0}
    death = install_death_receipt(death_path, lambda: deepcopy(progress))
    os.environ.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR", None)
    os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW"] = str(SUPERVISOR_WINDOW)
    os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    os.environ.update(
        {
            "CARNOT_ARC_GGUF_PATH": model["model_path"],
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(device["index"]),
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_SEED": str(cell["seed"]),
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_KV_QUANT": "q8_0",
            "CARNOT_ARC_INDUCE_N_CTX": str(CONTEXT_REQUESTED),
        }
    )
    random.seed(int(cell["seed"]))
    np.random.seed(int(cell["seed"]) % (2**32 - 1))
    lease = lease_api.GpuLease.acquire(
        runtime_dir=cell_root / "lease",
        task_id=f"exp6776-{cell['cell_id']}",
        device_uuid=device["uuid"],
        expected_model=model["model_path"],
        vram_before_mb=int(device.get("memory_used_mb", 0)),
        ttl_s=3600,
    )
    proposer = None
    shard_receipts = []
    responses = {"count": 0, "predicted_tokens": 0}
    started = time.monotonic()
    result: JsonDict = {
        **cell,
        "status": "failed",
        "failure_class": "worker_not_completed",
        "solve_claim": False,
    }
    try:
        lease.transition("admitted")
        lease.transition("loading")
        proposer = LocalGGUFProposer(
            repo_substr=model["repo_substr"],
            model_path=model["model_path"],
            n_ctx=CONTEXT_REQUESTED,
            max_tokens=int(model["decode_tokens"]),
            timeout=1200,
            port=port,
            mtp=False,
            kv_quant="q8_0",
        )
        original_record = proposer._record_completion_diagnostics

        def _record(response: Mapping[str, Any]) -> None:
            responses["count"] += 1
            responses["predicted_tokens"] += int(
                (response.get("timings") or {}).get("predicted_n") or 0
            )
            original_record(response)

        proposer._record_completion_diagnostics = _record
        if proposer._ensure_server() is not True:
            raise RuntimeError("llama_server_failed_to_start")
        server_pid = getattr(proposer._proc, "pid", None)
        resident = admission._gpu_snapshot(device["uuid"], server_pid)
        resident_owned_vram = int(resident.get("owned_pid_vram_mb", 0) or 0)
        lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0)))
        lease.transition("inferencing")
        policy = E3AgentPolicy(
            cell["game"], proposer=proposer, frontier_discipline_seed=int(cell["seed"])
        )
        if cell["supervisor_observation"] == "off":
            policy._trajectory_supervisor = None
            policy._trajectory_supervisor_applies = False
        original_supervise = policy._maybe_supervise_trajectory
        last_firings = 0

        def _checkpointed_supervise(latest: Any) -> None:
            nonlocal last_firings
            original_supervise(latest)
            progress["actions"] += 1
            receipt = policy.trajectory_supervisor_diagnostics()
            redirects = receipt.get("would_have_redirects") or []
            progress["firings"] = len(redirects)
            event = None
            if len(redirects) > last_firings:
                event = "supervisor_firing"
                last_firings = len(redirects)
            elif progress["actions"] % ACTION_BLOCK == 0:
                event = "action_block"
            if event:
                shard_receipts.append(
                    write_progress_shard(
                        shard_path,
                        cell_id=cell["cell_id"],
                        action_index=progress["actions"],
                        event=event,
                        trajectory_supervisor=receipt,
                    )
                )

        policy._maybe_supervise_trajectory = _checkpointed_supervise
        run_row = leaderboard.run_game(
            cell["game"], policy, budget=int(cell["action_budget"]), variant=0, reflect=None
        )
        receipt = policy.trajectory_supervisor_diagnostics()
        actions = [
            deepcopy(frame.get("move") or {}) for frame in run_row.get("frame_sequence") or []
        ]
        shard_receipts.append(
            write_progress_shard(
                shard_path,
                cell_id=cell["cell_id"],
                action_index=int(run_row.get("actions") or len(actions)),
                event="cell_complete",
                trajectory_supervisor=receipt,
            )
        )
        props = proposer.server_props() or {}
        result = {
            **cell,
            "status": "complete",
            "model_path": model["model_path"],
            "model_sha256": model["model_sha256"],
            "llama_server_process": {
                "pid": server_pid,
                "exit_code": None,
                "absent_after_exit": False,
            },
            "llama_server_log": str(getattr(proposer, "_stderr_log_path", "")),
            "live_model_invoked": responses["count"] > 0,
            "first_token_observed": responses["predicted_tokens"] > 0,
            "context_observed": proposer.observed_n_ctx() or props.get("n_ctx"),
            "trajectory_supervisor": receipt,
            "scored_action_hash": scored_action_hash(actions),
            "scored_actions": int(run_row.get("actions") or 0),
            "levels": int(run_row.get("levels") or 0),
            "death_receipt": death,
            "shard_receipts": shard_receipts,
            "duration_s": round(time.monotonic() - started, 6),
            "failure_class": None,
            "solve_claim": False,
        }
    except Exception as exc:
        result["failure_class"] = f"{type(exc).__name__}:{exc}"[:500]
        result["death_receipt"] = death
        result["shard_receipts"] = shard_receipts
    finally:
        server_pid = getattr(getattr(proposer, "_proc", None), "pid", None) if proposer else None
        before_used = int(device.get("memory_used_mb", 0))
        if lease.document.get("phase") in {"resident", "inferencing"}:
            lease.transition("unloading")
        if proposer is not None:
            proposer.stop()
        recovery, snapshot = admission._wait_for_vram_recovery(
            device["uuid"], server_pid or 0, before_used
        )
        if lease.document.get("phase") == "unloading":
            lease.transition(
                "validating",
                vram_mb=int(snapshot.get("memory_used_mb", 0)),
                exit_code=0,
                unload_observed=not bool(snapshot.get("owned_pid_present")),
            )
        lease.transition(
            "terminal_complete" if result.get("status") == "complete" else "terminal_blocked"
        )
        release = lease.release()
        with socket.socket() as probe:
            port_closed = probe.connect_ex(("127.0.0.1", port)) != 0
        result["llama_server_process"] = {
            "pid": server_pid,
            "exit_code": getattr(getattr(proposer, "_proc", None), "returncode", None),
            "absent_after_exit": not admission._gpu_snapshot(device["uuid"], server_pid).get(
                "owned_pid_present", False
            ),
        }
        log_path = getattr(proposer, "_stderr_log_path", None) if proposer else None
        try:
            log_text = Path(log_path).read_text(errors="replace") if log_path else ""
        except OSError:
            log_text = ""
        layers = admission.exp6752._gpu_layers_from_log(log_text, 999)
        result["gpu_receipt"] = {
            "device_uuid": device["uuid"],
            "gpu_layers": layers,
            "peak_vram_mb": resident_owned_vram if "resident_owned_vram" in locals() else 0,
            "lease_owner": lease.owner_receipt(),
            "lease_release": release,
            "vram_recovery": recovery,
            "port_release": {"port": port, "closed": port_closed},
            "unrelated_processes_signaled": [],
        }
        result["teardown_passed"] = not teardown_errors(
            {**result, "worker_process": {"exit_code": 0, "absent_after_exit": True}}
        )
        result["canonical_shard_path"] = str(shard_path)
        write_json_atomic(output_path, result)
        write_json_atomic(shard_path, {"rows": [result]})
    return 0 if result.get("status") == "complete" else 2


def run_refinement_cli(
    *, inputs: Sequence[Path], ledger_path: Path = LEDGER_PATH
) -> JsonDict:  # pragma: no cover - mutating CLI boundary
    """Invoke the public refinement script and retain its exact ledger result."""

    before = refinement.load_ledger(ledger_path)
    before_ids = sorted(before["entries"])
    before_hash = sha256_file(ledger_path) if ledger_path.is_file() else sha256_json(before)
    before_rec = refinement.evaluate(before, datetime.now(UTC).isoformat(timespec="seconds"))
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/arc_supervisor_refine.py"),
        "--ledger",
        str(ledger_path),
        "--json",
        *[str(path) for path in inputs],
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    after = refinement.load_ledger(ledger_path)
    recommendation = after.get("recommendation") or refinement.evaluate(
        after, datetime.now(UTC).isoformat(timespec="seconds")
    )
    after_ids = sorted(after["entries"])
    counts = recommendation.get("ingest_counts") or {}
    return {
        "tool": "scripts/arc_supervisor_refine.py",
        "command": command,
        "ran": True,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-16000:],
        "stderr": completed.stderr[-16000:],
        "inputs": [str(path) for path in inputs],
        "ledger_path": str(ledger_path),
        "ledger_sha256_before": before_hash,
        "ledger_sha256_after": sha256_file(ledger_path),
        "entry_ids_before": before_ids,
        "entry_ids_after": after_ids,
        "deduplicated": len(after_ids) == len(before_ids) + int(counts.get("applied_new") or 0),
        "ingest_counts": counts,
        "firings_before_by_arm": {row["arm"]: row["fired"] for row in before_rec["per_arm"]},
        "recommendation": recommendation,
    }


def run(
    *,
    result_path: Path = RESULT_PATH,
    manifest: Mapping[str, Any] | None = None,
    preflight_fn: Callable[[], JsonDict] = collect_preconditions,
    worker_runner: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Path], JsonDict
    ] = run_cell_subprocess,
    clock: Callable[[], int] = time.monotonic_ns,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run preconditions, isolated cells, refinement, validation, and atomic output."""

    started_ns = clock()
    frozen = deepcopy(dict(manifest or freeze_manifest()))
    preflight = preflight_fn()
    models = [deepcopy(dict(row)) for row in preflight.get("models", [])]
    if preflight.get("all_passed") is True:  # pragma: no cover - admitted live host
        selected = preflight.get("device_selection_receipt", {}).get("selected_device")
        by_id = {str(row.get("model_id")): row for row in models}
        cells = [
            worker_runner(spec, by_id[str(spec["model_id"])], selected, WORK_DIR)
            for spec in frozen["cells"]
        ]
        shard_inputs = [Path(str(row["canonical_shard_path"])) for row in cells]
        refinement_receipt = run_refinement_cli(inputs=shard_inputs)
    else:
        failed = failed_preflight_check(preflight)
        cells = _blocked_cells(frozen, models, str(failed.get("check")))
        refinement_receipt = _not_run_refinement_receipt(
            datetime.now(UTC).isoformat(timespec="seconds")
        )
    finished_ns = clock()
    artifact = build_artifact(
        manifest=frozen,
        models=models,
        preflight=preflight,
        cells=cells,
        refinement_receipt=refinement_receipt,
        duration_s=(finished_ns - started_ns) / 1_000_000_000,
        run_date=run_date,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp6776 artifact:" + ",".join(errors))
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - public CLI
    """Run the parent command or one explicitly scoped worker process."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_job is not None:
        if args.worker_output is None:
            parser.error("--worker-job requires --worker-output")
        return _worker_entry(args.worker_job, args.worker_output)
    if args.validate:
        artifact = json.loads(args.result_path.read_text())
        errors = validate_artifact(artifact)
        if errors:
            print("\n".join(errors))
            return 1
        print("OK")
        return 0
    artifact = run(result_path=args.result_path, run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(args.result_path),
                "ready": artifact["shadow_supervisor_transport_ready"],
                "verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
