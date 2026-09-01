"""Freeze terminal live ARC evidence strata without touching live runs.

Spec refs: REQ-ARC-6843 and SCENARIO-ARC-6843-*.

The inventory is an audit artifact. It reads terminal files, records hashes,
and samples process state with `ps`. It does not wait for a run to finish, send
signals, change leases, delete locks, or convert a partial run into evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-agi/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6843_live_arc_evidence_stratum_freeze.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6843_live_arc_evidence_stratum_freeze.py")
OUTPUT_PATH = Path("results/experiment_6843_live_arc_evidence_stratum_freeze.json")
SCHEMA = "carnot.experiment_6843.live_arc_evidence_stratum_freeze.v1"
INFERENCE_SUBSTRATE = "read_only_live_artifact_inventory"
READ_ONLY_PROCESS_COMMANDS = ["ps -eo pid,ppid,lstart,stat,etime,args --sort=pid"]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "process_observations",
    "reproducibility_checksum",
    "rows",
    "terminal_artifact_manifest",
    "incomplete_artifact_manifest",
    "configuration_strata",
    "supervisor_eligible_cells",
    "tool_gap_eligible_cells",
    "unmatched_cell_reasons",
    "arc_inventory_complete_score",
    "supervisor_cells_ready_score",
    "tool_gap_cells_ready_score",
    "solve_claim",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

TOP_LEVEL_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    *REQUIRED_ARTIFACT_FIELDS,
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "The schema lets downstream readers reject incompatible inventories.",
    "experiment_id": "The identifier binds the file to REQ-ARC-6843.",
    "run_date": "The date fixes this execution-time inventory.",
    "status": "The status separates a complete inventory from a blocked gate.",
    "field_principles": "Each top-level field states why an auditor needs it.",
    "preconditions_checked": "The gates stop source drift before row eligibility is counted.",
    "inference_substrate": "The substrate says this is a read-only inventory, not inference.",
    "duration_s": "Wall time exposes interrupted or skipped execution.",
    "source_artifact_hashes": "File hashes bind every terminal or missing source observation.",
    "process_observations": "Live processes are recorded as observations, not evidence rows.",
    "reproducibility_checksum": "One digest binds the non-timing inventory content.",
    "rows": "One row per run and game keeps configurations separated.",
    "terminal_artifact_manifest": "Terminal files are named without trusting their claims.",
    "incomplete_artifact_manifest": "Partial, missing, and unreadable files are ineligible.",
    "configuration_strata": "Distinct model, policy, budget, game, tool, and supervisor strata stay separate.",
    "supervisor_eligible_cells": "The exact supervisor receipt count drives readiness only.",
    "tool_gap_eligible_cells": "The exact tool-gap receipt count drives readiness only.",
    "unmatched_cell_reasons": "Ineligible cells remain visible with a reason.",
    "arc_inventory_complete_score": "A complete score means provenance is inventoried, not that ARC was solved.",
    "supervisor_cells_ready_score": "A count-derived readiness flag makes no effect claim.",
    "tool_gap_cells_ready_score": "A count-derived readiness flag makes no effect claim.",
    "solve_claim": "False prevents an inventory from becoming a game solve claim.",
    "gate_check_summary": "A failed gate names the check, expected value, and observed value.",
    "verifier_is_oracle": "False states that this audit does not adjudicate solutions.",
    "verdict_class": "A closed verdict class prevents unsupported positives.",
    "honest_verdict": "The terminal complete_ prefix states the evidence boundary.",
}

MODEL_BY_REPO_SUBSTR = {
    "Qwen3.8-27B": "unsloth/Qwen3.8-27B-GGUF",
    "Qwen3.6-35B-A3B": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gemma-4-31B": "unsloth/gemma-4-31B-it-GGUF",
    "gemma-4-26B-A4B": "unsloth/gemma-4-26B-A4B-it-GGUF",
}
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
TERMINAL_CLASSES = {
    "complete_terminal",
    "blocked_terminal",
    "disqualified_terminal",
    "partial_terminal",
}


def canonical_json(value: Any) -> str:
    """Use one JSON form so checksums are stable across rebuilds."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return a prefixed digest. Prefixes stop algorithm ambiguity later."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible content through the canonical encoder."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file by streaming. Some ARC artifacts contain large traces."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_bytes(path: Path) -> tuple[bytes, str | None]:
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return b"", type(exc).__name__


def _json_object(raw: bytes) -> JsonDict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_default_source_paths(root: Path = REPO_ROOT) -> dict[str, Path]:
    """Collect the bounded source set for this inventory.

    The list is explicit for gate inputs and known mechanism artifacts. The
    leaderboard run directory is globbed because each completed subset run uses
    a process-id filename.
    """

    paths: dict[str, Path] = {
        "v598_root": root / "results/experiment_6835_v598_terminal_evidence_freeze.json",
        "arc_registry": root / "ops/arc_solve_registry.yaml",
        "ops_status": root / "ops/status.md",
        "canonical_path_doc": root / "ops/arc-live-agent-canonical-path.md",
        "arc_competition_agent_source": root / "python/carnot/agentic/arc_competition_agent.py",
        "arc_trajectory_supervisor_source": root
        / "python/carnot/agentic/arc_trajectory_supervisor.py",
        "arc_leaderboard_eval_source": root / "scripts/arc_leaderboard_eval.py",
        "experiment_6843_module_source": root / MODULE_PATH,
        "experiment_6843_wrapper_source": root / WRAPPER_PATH,
        "upstream_exp6681": root / "results/experiment_6681_arc_post_redirect_outcomes.json",
        "upstream_exp6776": root / "results/experiment_6776_arc_shadow_supervisor_accrual.json",
        "upstream_exp6777": root / "results/experiment_6777_arc_tool_gap_transport.json",
        "upstream_exp6820_missing": root
        / "results/experiment_6820_arc_tool_gap_obligation_transport_v2.json",
        "supervisor_ledger": root / "ops/arc_supervisor_refinement_ledger.json",
        "tool_gap_ledger_missing": root / "ops/arc_tool_gap_ledger.json",
        "supervisor_exp6524": root
        / "results/experiment_6524_arc_supervisor_redirect_generalization.json",
        "supervisor_exp6682": root / "results/experiment_6682_arc_held_family_supervisor_ab.json",
        "supervisor_exp6656": root / "results/experiment_6656_arc_trace_automaton_live_loo.json",
        "supervisor_exp6558": root
        / "results/experiment_6558_arc_live_redirect_ledger_reachability.json",
    }
    run_dir = root / "results/arc_leaderboard_eval_runs"
    if run_dir.exists():
        for path in sorted(run_dir.glob("*.json")):
            paths[f"leaderboard:{path.name}"] = path
    return paths


def _terminal_class(path: Path, payload: Mapping[str, Any], read_error: str | None) -> str:
    if read_error is not None:
        return "missing_artifact" if read_error == "FileNotFoundError" else "unreadable_artifact"
    status = str(payload.get("status", "")).lower()
    verdict = str(payload.get("honest_verdict", "")).lower()
    verdict_class = str(payload.get("verdict_class", "")).lower()
    schema = str(payload.get("schema", "")).lower()
    if path.name.endswith(".partial.json") or payload.get("complete") is False:
        return "incomplete_artifact"
    if schema == "carnot.arc.supervisor_refinement_ledger.v1":
        return "complete_terminal"
    if "blocked" in status or verdict.startswith("blocked") or verdict_class == "blocked":
        return "blocked_terminal"
    if "disqualified" in status or verdict_class == "disqualified":
        return "disqualified_terminal"
    if "partial" in status or verdict.startswith("partial_") or verdict_class == "partial":
        return "partial_terminal"
    if (
        payload.get("complete") is True
        or status == "complete"
        or status.startswith("complete_")
        or verdict.startswith("complete_")
    ):
        return "complete_terminal"
    return "incomplete_artifact"


def _artifact_family(name: str, path: Path) -> str:
    text = f"{name}:{path.as_posix()}"
    if name.endswith("_source") or name in {
        "arc_registry",
        "ops_status",
        "canonical_path_doc",
        "v598_root",
    }:
        return "precondition_source"
    if "arc_leaderboard_eval_runs" in text:
        return "leaderboard_trajectory"
    if "arc_supervisor_refinement_ledger" in text:
        return "supervisor_refinement_ledger"
    if "tool_gap" in text or "6777" in text or "6820" in text:
        return "tool_gap_artifact"
    if "supervisor" in text or "6656" in text or "6558" in text or "6524" in text:
        return "supervisor_artifact"
    return "source_artifact"


def _producer_configuration(payload: Mapping[str, Any], family: str) -> JsonDict:
    config: JsonDict = {
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": payload.get("verdict_class"),
    }
    for key in ("experiment", "games_mode", "policy", "budget", "random_seed", "complete"):
        if key in payload:
            config[key] = payload.get(key)
    if family == "leaderboard_trajectory":
        config["per_game_count"] = len(payload.get("per_game", []))
        config["games"] = [row.get("game") for row in payload.get("per_game", [])]
    if "supervisor" in family:
        for key in ("supervisor_mode", "supervisor_window", "shadow_supervisor_transport_ready"):
            if key in payload:
                config[key] = payload.get(key)
    if "model_specs" in payload:
        config["model_specs_sha256"] = sha256_json(payload["model_specs"])
    return config


def _source_record(name: str, path: Path, root: Path) -> JsonDict:
    raw, read_error = _read_bytes(path)
    payload = _json_object(raw)
    family = _artifact_family(name, path)
    terminal_class = _terminal_class(path, payload, read_error)
    return {
        "name": name,
        "path": _relative(path, root),
        "exists": read_error is None,
        "read_error": read_error,
        "file_sha256": sha256_bytes(raw) if read_error is None else None,
        "size_bytes": len(raw) if read_error is None else None,
        "artifact_family": family,
        "terminal_class": terminal_class,
        "producer_configuration": _producer_configuration(payload, family),
        "_payload": payload,
        "_raw_text": raw.decode("utf-8", errors="replace") if read_error is None else "",
    }


def _source_records(source_paths: Mapping[str, Path], root: Path) -> dict[str, JsonDict]:
    return {name: _source_record(name, path, root) for name, path in sorted(source_paths.items())}


def _check(check: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def evaluate_preconditions(records: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Check only the requested gates. Missing optional artifacts stay inventory rows."""

    v598 = records.get("v598_root", {}).get("_payload", {})
    canonical = records.get("canonical_path_doc", {}).get("_payload", {})
    canonical_record = records.get("canonical_path_doc", {})
    canonical_text = str(canonical_record.get("_raw_text", "")) if not canonical else ""

    source_identity = {
        "canonical_doc_readable": records.get("canonical_path_doc", {}).get("exists") is True,
        "canonical_doc_names_entrypoint": "make_carnot_agent" in canonical_text
        and "E3AgentPolicy" in canonical_text,
        "agent_source_readable": records.get("arc_competition_agent_source", {}).get("exists")
        is True,
        "supervisor_source_readable": records.get("arc_trajectory_supervisor_source", {}).get(
            "exists"
        )
        is True,
        "agent_source_hash": records.get("arc_competition_agent_source", {}).get("file_sha256"),
        "supervisor_source_hash": records.get("arc_trajectory_supervisor_source", {}).get(
            "file_sha256"
        ),
    }
    source_expected = {
        "canonical_doc_readable": True,
        "canonical_doc_names_entrypoint": True,
        "agent_source_readable": True,
        "supervisor_source_readable": True,
        "agent_source_hash": source_identity["agent_source_hash"],
        "supervisor_source_hash": source_identity["supervisor_source_hash"],
    }
    return [
        _check("v598_evidence_root_ready_score", 1, v598.get("v598_evidence_root_ready_score")),
        _check("arc_registry_readable", True, records.get("arc_registry", {}).get("exists")),
        _check("ops_status_readable", True, records.get("ops_status", {}).get("exists")),
        _check("canonical_path_source_identity", source_expected, source_identity),
    ]


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = next((row for row in checks if not row.get("passed")), None)
    if failed is None:
        return {
            "passed": True,
            "failed_check": None,
            "expected": "all gates pass",
            "observed": "all gates pass",
        }
    return {
        "passed": False,
        "failed_check": failed.get("check"),
        "expected": failed.get("expected"),
        "observed": failed.get("observed"),
    }


def _extract_model_id(game_row: Mapping[str, Any]) -> str:
    diagnostics = game_row.get("policy_diagnostics", {})
    proposer = diagnostics.get("proposer", {}) if isinstance(diagnostics, Mapping) else {}
    repo_substr = proposer.get("repo_substr") if isinstance(proposer, Mapping) else None
    if isinstance(repo_substr, str):
        return MODEL_BY_REPO_SUBSTR.get(repo_substr, f"unsloth/{repo_substr}-GGUF")
    attempts = diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
    for attempt in attempts if isinstance(attempts, list) else []:
        model_specs = str(attempt.get("model_specs", ""))
        for marker, model_id in MODEL_BY_REPO_SUBSTR.items():
            if marker in model_specs:
                return model_id
    return "unknown"


def _tool_gap_summary(game_row: Mapping[str, Any]) -> JsonDict:
    diagnostics = game_row.get("policy_diagnostics", {})
    attempts = diagnostics.get("induction_attempts", []) if isinstance(diagnostics, Mapping) else []
    receipts: list[Mapping[str, Any]] = []
    for attempt in attempts if isinstance(attempts, list) else []:
        if isinstance(attempt, Mapping) and isinstance(attempt.get("tool_gap"), Mapping):
            receipts.append(attempt["tool_gap"])
    calls = sum(int(row.get("tool_calls_total", 0) or 0) for row in receipts)
    events = sum(len(row.get("tool_gap_events", []) or []) for row in receipts)
    complete = all(
        set(row)
        >= {
            "tool_gap_events",
            "tool_gap_events_dropped",
            "candidate_tools_enabled",
            "candidate_tools_rejected",
            "terminated_by",
            "tool_calls_total",
        }
        for row in receipts
    )
    enabled = sorted(
        {
            str(item)
            for row in receipts
            for item in row.get("candidate_tools_enabled", [])
            if isinstance(item, str)
        }
    )
    return {
        "state": "selfparse" if receipts else "off_or_unobserved",
        "receipt_complete": bool(receipts) and complete,
        "receipt_count": len(receipts),
        "tool_calls_total": calls,
        "tool_gap_event_count": events,
        "candidate_tools_enabled": enabled,
    }


def _supervisor_summary(game_row: Mapping[str, Any]) -> JsonDict:
    diagnostics = game_row.get("policy_diagnostics", {})
    receipt = diagnostics.get("trajectory_supervisor") if isinstance(diagnostics, Mapping) else None
    if not isinstance(receipt, Mapping):
        return {"state": "unobserved", "receipt_complete": False, "receipt_hash": None}
    mode = "applied" if receipt.get("mode") == "applied" else "shadow"
    complete = "enabled" in receipt and ("redirects" in receipt or "arm_outcomes" in receipt)
    return {"state": mode, "receipt_complete": complete, "receipt_hash": sha256_json(receipt)}


def _row_with_hash(row: JsonDict) -> JsonDict:
    material = {key: value for key, value in row.items() if key != "row_sha256"}
    return {**row, "row_sha256": sha256_json(material)}


def _stratum_identity(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(key))
        for key in (
            "artifact_family",
            "run_id",
            "game",
            "model_id",
            "policy",
            "budget",
            "tool_loop_state",
            "supervisor_state",
            "receipt_completeness",
        )
    )


def _leaderboard_rows(record: Mapping[str, Any]) -> list[JsonDict]:
    payload = record["_payload"]
    rows: list[JsonDict] = []
    for game_row in payload.get("per_game", []):
        if not isinstance(game_row, Mapping) or not game_row.get("game"):
            continue
        tool = _tool_gap_summary(game_row)
        supervisor = _supervisor_summary(game_row)
        frame_sequence = game_row.get("frame_sequence", [])
        trajectory_complete = isinstance(frame_sequence, list) and isinstance(
            game_row.get("actions"), int
        )
        row: JsonDict = {
            "artifact_family": "leaderboard_trajectory",
            "source_path": record["path"],
            "source_artifact_sha256": record["file_sha256"],
            "run_id": Path(str(record["path"])).name.removesuffix(".json").removesuffix(".partial"),
            "game": str(game_row["game"]),
            "model_id": _extract_model_id(game_row),
            "policy": payload.get("policy", "unknown"),
            "budget": payload.get("budget"),
            "tool_loop_state": tool["state"],
            "supervisor_state": supervisor["state"],
            "receipt_completeness": "complete",
            "trajectory_receipt_complete": trajectory_complete,
            "trajectory_frame_count": len(frame_sequence)
            if isinstance(frame_sequence, list)
            else 0,
            "trajectory_sha256": sha256_json(frame_sequence)
            if isinstance(frame_sequence, list)
            else None,
            "supervisor_receipt_complete": supervisor["receipt_complete"],
            "supervisor_receipt_sha256": supervisor["receipt_hash"],
            "tool_gap_receipt_complete": tool["receipt_complete"],
            "tool_gap_receipt_count": tool["receipt_count"],
            "tool_gap_calls_total": tool["tool_calls_total"],
            "tool_gap_event_count": tool["tool_gap_event_count"],
            "producer_configuration": {
                "games_mode": payload.get("games_mode"),
                "random_seed": payload.get("random_seed"),
                "complete": payload.get("complete"),
                "levels_recorded": game_row.get("levels"),
                "actions_recorded": game_row.get("actions"),
                "charged_actions": game_row.get("charged_actions"),
            },
            "eligibility": {
                "terminal_artifact": True,
                "run_game_complete": True,
                "mechanism_effect_claim": False,
            },
        }
        row["stratum_identity"] = _stratum_identity(row)
        rows.append(_row_with_hash(row))
    return rows


def _supervisor_ledger_rows(record: Mapping[str, Any]) -> list[JsonDict]:
    payload = record["_payload"]
    entries = payload.get("entries", {})
    rows: list[JsonDict] = []
    for key, entry in sorted(entries.items()):
        if not isinstance(entry, Mapping) or not entry.get("game"):
            continue
        receipt = {
            "receipt_id": entry.get("receipt_id", key),
            "mode": entry.get("mode"),
            "window": entry.get("window"),
            "redirects": entry.get("redirects", []),
        }
        row = {
            "artifact_family": "supervisor_refinement_ledger",
            "source_path": record["path"],
            "source_artifact_sha256": record["file_sha256"],
            "run_id": str(entry.get("receipt_id", key)),
            "game": str(entry["game"]),
            "model_id": "unknown",
            "policy": entry.get("harness_arm", "unknown"),
            "budget": entry.get("actions_observed"),
            "tool_loop_state": "unknown",
            "supervisor_state": str(entry.get("mode", "unknown")),
            "receipt_completeness": "complete",
            "trajectory_receipt_complete": False,
            "trajectory_frame_count": 0,
            "trajectory_sha256": None,
            "supervisor_receipt_complete": bool(entry.get("receipt_id")) and "redirects" in entry,
            "supervisor_receipt_sha256": sha256_json(receipt),
            "tool_gap_receipt_complete": False,
            "tool_gap_receipt_count": 0,
            "tool_gap_calls_total": 0,
            "tool_gap_event_count": 0,
            "producer_configuration": {
                "seed": entry.get("seed"),
                "levels_recorded": entry.get("levels"),
                "window": entry.get("window"),
                "recommendation_status": payload.get("recommendation", {}).get("status")
                if isinstance(payload.get("recommendation"), Mapping)
                else None,
            },
            "eligibility": {
                "terminal_artifact": True,
                "run_game_complete": True,
                "mechanism_effect_claim": False,
            },
        }
        row["stratum_identity"] = _stratum_identity(row)
        rows.append(_row_with_hash(row))
    return rows


def build_rows(records: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Extract run/game rows from terminal complete evidence only."""

    rows: list[JsonDict] = []
    for record in records.values():
        if record.get("terminal_class") != "complete_terminal":
            continue
        family = record.get("artifact_family")
        if family == "leaderboard_trajectory":
            rows.extend(_leaderboard_rows(record))
        elif family == "supervisor_refinement_ledger":
            rows.extend(_supervisor_ledger_rows(record))
    return rows


def _duplicate_identities(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    counts = Counter(row["stratum_identity"] for row in rows)
    return sorted(identity for identity, count in counts.items() if count > 1)


def _eligible_summary(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    eligible = [row for row in rows if row.get(field) is True]
    return {
        "count": len(eligible),
        "row_ids": [row["stratum_identity"] for row in eligible],
        "exact_count_rule": f"count rows where {field} is true",
    }


def _unmatched_reasons(
    rows: Sequence[Mapping[str, Any]], incomplete: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    reasons: list[JsonDict] = []
    for row in rows:
        missing = []
        if row.get("supervisor_receipt_complete") is not True:
            missing.append("supervisor_receipt_missing_or_unobserved")
        if row.get("tool_gap_receipt_complete") is not True:
            missing.append("tool_gap_receipt_missing_or_unobserved")
        if row.get("trajectory_receipt_complete") is not True:
            missing.append("trajectory_receipt_missing")
        if missing:
            reasons.append(
                {
                    "stratum_identity": row["stratum_identity"],
                    "source_path": row["source_path"],
                    "game": row["game"],
                    "reasons": missing,
                    "ineligible_for_effect_claim": True,
                }
            )
    for item in incomplete:
        reasons.append(
            {
                "stratum_identity": None,
                "source_path": item["path"],
                "game": None,
                "reasons": [item["terminal_class"]],
                "ineligible_for_effect_claim": True,
            }
        )
    return reasons


def _configuration_strata(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    strata = sorted(
        {
            "|".join(
                str(row.get(key))
                for key in (
                    "model_id",
                    "policy",
                    "budget",
                    "game",
                    "tool_loop_state",
                    "supervisor_state",
                )
            )
            for row in rows
        }
    )
    return {"count": len(strata), "strata": strata}


def _public_record(record: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in record.items() if key not in {"_payload", "_raw_text"}}


def source_artifact_hashes(records: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {name: _public_record(record) for name, record in records.items()}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact content except wall-clock duration and this field."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def _artifact_shell(
    *,
    run_date: str,
    duration_s: float,
    records: Mapping[str, Mapping[str, Any]],
    process_observations: Sequence[Mapping[str, Any]],
) -> JsonDict:
    inventory_records = [
        record
        for record in records.values()
        if record.get("artifact_family") != "precondition_source"
    ]
    terminal = [
        _public_record(record)
        for record in inventory_records
        if record.get("terminal_class") in TERMINAL_CLASSES
    ]
    incomplete = [
        _public_record(record)
        for record in inventory_records
        if record.get("terminal_class") not in TERMINAL_CLASSES
    ]
    return {
        "schema": SCHEMA,
        "experiment_id": "exp6843-live-arc-evidence-stratum-freeze",
        "run_date": run_date,
        "status": "complete_live_arc_inventory",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": source_artifact_hashes(records),
        "process_observations": list(process_observations),
        "reproducibility_checksum": "",
        "rows": [],
        "terminal_artifact_manifest": terminal,
        "incomplete_artifact_manifest": incomplete,
        "configuration_strata": {"count": 0, "strata": []},
        "supervisor_eligible_cells": {"count": 0, "row_ids": [], "exact_count_rule": ""},
        "tool_gap_eligible_cells": {"count": 0, "row_ids": [], "exact_count_rule": ""},
        "unmatched_cell_reasons": [],
        "arc_inventory_complete_score": 0,
        "supervisor_cells_ready_score": 0,
        "tool_gap_cells_ready_score": 0,
        "solve_claim": False,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "null",
        "honest_verdict": "complete_live_arc_inventory_terminal_evidence_only_no_solve_claim",
    }


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_paths: Mapping[str, Path] | None = None,
    process_observations: Sequence[Mapping[str, Any]] | None = None,
    root: Path | None = None,
) -> JsonDict:
    """Build the immutable inventory from file bytes and process observations."""

    if root is None:
        root = _common_root(source_paths) if source_paths is not None else REPO_ROOT
    paths = dict(source_paths or collect_default_source_paths(root))
    records = _source_records(paths, root)
    artifact = _artifact_shell(
        run_date=run_date,
        duration_s=duration_s,
        records=records,
        process_observations=process_observations or [],
    )
    checks = evaluate_preconditions(records)
    rows = build_rows(records)
    duplicates = _duplicate_identities(rows)
    if duplicates:
        checks.append(_check("duplicate_row_identities", [], duplicates))
    if not rows:
        checks.append(_check("terminal_run_game_rows_present", True, False))
    summary = _gate_summary(checks)
    artifact["preconditions_checked"] = checks
    artifact["gate_check_summary"] = summary
    if not summary["passed"]:
        artifact["status"] = "complete_blocked_live_arc_inventory"
        artifact["rows"] = []
        artifact["configuration_strata"] = {"count": 0, "strata": []}
        artifact["unmatched_cell_reasons"] = _unmatched_reasons(
            [], artifact["incomplete_artifact_manifest"]
        )
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = "complete_blocked_live_arc_inventory"
    else:
        supervisor = _eligible_summary(rows, "supervisor_receipt_complete")
        tool_gap = _eligible_summary(rows, "tool_gap_receipt_complete")
        artifact["rows"] = rows
        artifact["configuration_strata"] = _configuration_strata(rows)
        artifact["supervisor_eligible_cells"] = supervisor
        artifact["tool_gap_eligible_cells"] = tool_gap
        artifact["unmatched_cell_reasons"] = _unmatched_reasons(
            rows, artifact["incomplete_artifact_manifest"]
        )
        artifact["supervisor_cells_ready_score"] = int(supervisor["count"] > 0)
        artifact["tool_gap_cells_ready_score"] = int(tool_gap["count"] > 0)
        artifact["arc_inventory_complete_score"] = int(
            bool(rows)
            and not duplicates
            and all(str(row.get("row_sha256", "")).startswith("sha256:") for row in rows)
            and all(item.get("reasons") for item in artifact["unmatched_cell_reasons"])
        )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _common_root(source_paths: Mapping[str, Path] | None) -> Path:
    if not source_paths:
        return REPO_ROOT
    existing = [path.resolve() for path in source_paths.values() if path.exists()]
    if not existing:
        return REPO_ROOT
    return Path(os.path.commonpath([str(path) for path in existing]))


def extract_wrapper_configuration(text: str) -> JsonDict:
    supervisor = (
        "applied"
        if re.search(r"CARNOT_ARC_TRAJECTORY_SUPERVISOR[\"']\s*:\s*[\"']1", text)
        else "unobserved"
    )
    tool_match = re.search(r"CARNOT_ARC_INDUCE_TOOL_LOOP[\"']\s*:\s*[\"']([^\"']+)", text)
    return {
        "supervisor_state": supervisor,
        "tool_loop_state": tool_match.group(1) if tool_match else "off_or_unobserved",
    }


def _command_config(command: str) -> JsonDict:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    config: JsonDict = {
        "budget": None,
        "games": [],
        "model_id": None,
        "policy": None,
        "supervisor_state": "unobserved",
        "tool_loop_state": "off_or_unobserved",
    }
    for index, part in enumerate(parts):
        if part == "--policy" and index + 1 < len(parts):
            config["policy"] = parts[index + 1]
        if part == "--only" and index + 1 < len(parts):
            config["games"] = [item for item in parts[index + 1].split(",") if item]
        if part == "--budget" and index + 1 < len(parts):
            try:
                config["budget"] = int(parts[index + 1])
            except ValueError:
                config["budget"] = parts[index + 1]
        if part == "-m" and index + 1 < len(parts):
            model_path = Path(parts[index + 1])
            config["model_id"] = _model_id_from_path(model_path)
    return config


def _model_id_from_path(path: Path) -> str:
    text = path.as_posix()
    for marker, model_id in MODEL_BY_REPO_SUBSTR.items():
        if marker in text:
            return model_id
    return path.name or "unknown"


def parse_ps_line(line: str) -> JsonDict:
    """Parse one `ps` row from the read-only process sampler."""

    match = re.match(
        r"^\s*(?P<pid>\d+)\s+(?P<ppid>\d+)\s+"
        r"(?P<start>\w+\s+\w+\s+\d+\s+\d+:\d+:\d+\s+\d+)\s+"
        r"(?P<stat>\S+)\s+(?P<elapsed>\S+)\s+(?P<cmd>.*)$",
        line,
    )
    if not match:
        raise ValueError(f"unparseable ps line: {line!r}")
    command = match.group("cmd")
    return {
        "pid": int(match.group("pid")),
        "ppid": int(match.group("ppid")),
        "start_time": match.group("start"),
        "state": match.group("stat"),
        "elapsed": match.group("elapsed"),
        "command": command,
        "observed_configuration": _command_config(command),
        "read_only_commands": list(READ_ONLY_PROCESS_COMMANDS),
        "observation_role": "in_flight_process",
    }


def _read_proc_environ(pid: int) -> list[str]:
    path = Path("/proc") / str(pid) / "environ"
    raw, error = _read_bytes(path)
    if error is not None:
        return []
    return [item for item in raw.decode("utf-8", errors="replace").split("\x00") if item]


def _overlay_env_config(observation: JsonDict) -> JsonDict:
    env = _read_proc_environ(int(observation["pid"]))
    by_name = dict(item.split("=", 1) for item in env if "=" in item)
    config = dict(observation["observed_configuration"])
    if by_name.get("CARNOT_ARC_TRAJECTORY_SUPERVISOR") == "1":
        config["supervisor_state"] = "applied"
    if by_name.get("CARNOT_ARC_INDUCE_TOOL_LOOP"):
        config["tool_loop_state"] = by_name["CARNOT_ARC_INDUCE_TOOL_LOOP"]
    observation["observed_configuration"] = config
    observation["observed_env_keys"] = sorted(
        key for key in by_name if key.startswith("CARNOT_ARC_") or key == "CUDA_VISIBLE_DEVICES"
    )
    return observation


def sample_process_observations() -> list[JsonDict]:
    """Run one read-only `ps` sample and parse ARC-related live processes."""

    completed = subprocess.run(
        READ_ONLY_PROCESS_COMMANDS[0].split(),
        check=False,
        capture_output=True,
        text=True,
    )
    observations: list[JsonDict] = []
    for line in completed.stdout.splitlines()[1:]:
        if not re.search(r"arc_leaderboard_eval|llama-server|supervised_run\.py", line):
            continue
        try:
            observations.append(_overlay_env_config(parse_ps_line(line)))
        except ValueError:
            continue
    return observations


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema errors. The producer fails closed when this is nonempty."""

    errors: list[str] = []
    missing = [field for field in TOP_LEVEL_FIELDS if field not in artifact]
    if missing:
        errors.append("required artifact fields are missing")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field principles do not cover every top-level field")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim must be false")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest verdict lacks complete_ terminal prefix")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    duplicate_rows = _duplicate_identities(artifact.get("rows", []))
    if duplicate_rows:
        errors.append("duplicate row identities present")
    if artifact.get("status") == "complete_blocked_live_arc_inventory":
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked verdict_class mismatch")
        if artifact.get("rows") != []:
            errors.append("blocked artifact emitted rows")
        if artifact.get("arc_inventory_complete_score") != 0:
            errors.append("blocked artifact marked complete")
        if not artifact.get("gate_check_summary", {}).get("failed_check"):
            errors.append("blocked artifact lacks failed check")
    elif artifact.get("status") == "complete_live_arc_inventory":
        if artifact.get("verdict_class") != "null":
            errors.append("complete inventory verdict_class mismatch")
        if artifact.get("arc_inventory_complete_score") != 1:
            errors.append("inventory complete score mismatch")
        if not artifact.get("rows"):
            errors.append("complete inventory lacks rows")
    else:
        errors.append("status mismatch")
    return errors


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze live ARC evidence strata.")
    parser.add_argument("--date", required=True, help="execution date as YYYYMMDD")
    parser.add_argument("--output", default=str(REPO_ROOT / OUTPUT_PATH), help="artifact path")
    args = parser.parse_args(argv)

    start = time.monotonic()
    paths = collect_default_source_paths(REPO_ROOT)
    process_observations = sample_process_observations()
    duration_s = time.monotonic() - start
    artifact = build_artifact(
        run_date=args.date,
        duration_s=duration_s,
        source_paths=paths,
        process_observations=process_observations,
        root=REPO_ROOT,
    )
    errors = validate_artifact(artifact)
    if errors:
        for error in errors:
            print(f"validation error: {error}", file=sys.stderr)
        return 1
    _write_json_atomic(Path(args.output), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
