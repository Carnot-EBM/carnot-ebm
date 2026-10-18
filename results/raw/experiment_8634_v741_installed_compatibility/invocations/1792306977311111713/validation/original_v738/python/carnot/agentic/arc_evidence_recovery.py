"""Cold, read-only reduction of Exp7709 ARC evidence (REQ-ARC-WMTE-7722)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]


def authenticate_bytes(path: Path, expected: str | None, upstream: str, field: str) -> Json:
    """Compare exact bytes to the original owner hash and retain both operands."""
    observed = sha256_file(path) if path.is_file() else None
    return {
        "check": "historical_input_bytes",
        "upstream": upstream,
        "path": str(path),
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed is not None and observed == expected,
    }


def classify_response(response: Mapping[str, Any]) -> Json:
    """Diagnose acceptance from the original model bytes, including hidden reasoning."""
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    finish = choice.get("finish_reason")
    reason = (
        "reasoning_only_truncated_no_engine_code"
        if finish == "length" and not content and reasoning
        else "missing_final_engine_code"
        if not content
        else "final_content_present_unverified"
    )
    return {
        "finish_reason": finish,
        "final_content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "completion_tokens": int(response.get("usage", {}).get("completion_tokens") or 0),
        "acceptance_reason": reason,
    }


def join_episode_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Join actions to observations and reject missing or duplicate SDK identities."""
    if [row.get("game") for row in rows] != ["wa30", "lf52"]:
        raise ValueError("frozen_game_identity")
    joined = 0
    for row in rows:
        observations = {item["observation_id"]: item for item in row["observations"]}
        if len(observations) != len(row["observations"]):
            raise ValueError("duplicate_observation")
        for index, action in enumerate(row["actions"], 1):
            observation = observations.get(action.get("observation_id"))
            if observation is None or action.get("action_index") != index:
                raise ValueError("observation_join")
            if int(action["level_after"]) != int(observation["level"]):
                raise ValueError("level_observation_mismatch")
            joined += 1
    return {"joined_actions": joined, "observed_games": len(rows)}


def summarize_supervisor(receipts: Sequence[Mapping[str, Any]]) -> Json:
    """Count unique traces once and retain zero-firing outcomes as measured zeros."""
    unique: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        unique.setdefault(str(receipt["trace_id"]), receipt)
    per_game: Json = {}
    for receipt in unique.values():
        game = str(receipt["game"])
        game_row = per_game.setdefault(
            game,
            {
                "firings": 0,
                "applied_redirections": 0,
                "resolved_by_levelup": 0,
                "actions_to_levelup": [],
            },
        )
        game_row["firings"] += int(receipt["firings"])
        game_row["applied_redirections"] += int(receipt.get("applied_redirections", 0))
        game_row["resolved_by_levelup"] += int(receipt["resolved_by_levelup"])
        game_row["actions_to_levelup"].extend(receipt["actions_to_levelup"])
    return {
        "unique_traces": len(unique),
        "duplicate_traces": len(receipts) - len(unique),
        "per_game": per_game,
    }


def _load_json(path: Path, checks: list[Json], hashes: Json, field: str) -> Json | None:
    """Record missing custody before attempting a schema read."""
    observed = sha256_file(path) if path.is_file() else None
    checks.append(
        {
            "check": "historical_input_bytes",
            "upstream": "experiment_7709",
            "path": str(path),
            "field": field,
            "operator": "!=",
            "expected": None,
            "observed": observed,
            "passed": observed is not None,
        }
    )
    if observed is None:
        hashes["missing_custody"].append(str(path))
        return None
    hashes["flagged_historical_evidence"][str(path)] = observed
    return json.loads(path.read_text(encoding="utf-8"))


def _supervisor_receipt(path: Path, game: str, actions: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce only selection events emitted by the original scored E3 run."""
    selections = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"supervisor_arm_selection"' in line and '"selection"' in line
    ]
    selections = [
        row
        for row in selections
        if row.get("seam") == "supervisor_arm_selection" and row.get("event") == "selection"
    ]
    if len(selections) != len(actions):
        raise ValueError("supervisor_selection_count")
    firings = sum(row.get("supervisor_fired") is True for row in selections)
    applied = sum(row.get("applied_redirection") is True for row in selections)
    levelups = [
        int(row["action_index"])
        for row in actions
        if int(row["level_after"]) > int(row["level_before"])
    ]
    return {
        "trace_id": game + ":live",
        "game": game,
        "firings": firings,
        "applied_redirections": applied,
        "resolved_by_levelup": 0 if not firings else sum(bool(levelups) for _ in range(firings)),
        "actions_to_levelup": levelups,
        "selection_receipts": len(selections),
    }


def recover_history(root: Path, raw: Path) -> Json:
    """Authenticate and cold-join the two immutable V671 live attempts."""
    root, raw = root.resolve(), raw.resolve()
    checks: list[Json] = []
    hashes: Json = {
        "valid_producers": {},
        "flagged_historical_evidence": {},
        "pre_gate_receipts": {},
        "missing_custody": [],
    }
    old_path = root / "results/experiment_7709_v671_arc_first_contact.json"
    old = _load_json(old_path, checks, hashes, "original_artifact")
    rows_doc = _load_json(raw / "episode_rows.json", checks, hashes, "episode_rows")
    ledger_path = raw / "actions.jsonl"
    registry_path = root / "ops/arc_solve_registry.yaml"
    for path, field in ((ledger_path, "actions_jsonl"), (registry_path, "registry_yaml")):
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "check": "historical_input_bytes",
                "upstream": "experiment_7709",
                "path": str(path),
                "field": field,
                "operator": "!=",
                "expected": None,
                "observed": observed,
                "passed": observed is not None,
            }
        )
        if observed is None:
            hashes["missing_custody"].append(str(path))
        else:
            hashes["flagged_historical_evidence" if path == ledger_path else "valid_producers"][
                str(path)
            ] = observed
    if old is None or rows_doc is None or not ledger_path.is_file() or not registry_path.is_file():
        return {
            "failed_checks": [row for row in checks if not row["passed"]],
            "checks": checks,
            "hashes": hashes,
            "rows": [],
            "registry_precheck": {},
        }
    checks.append(
        {
            "check": "original_verdict",
            "upstream": "experiment_7709",
            "path": str(old_path),
            "field": "honest_verdict",
            "operator": "==",
            "expected": "complete_disqualified_required_validation",
            "observed": old.get("honest_verdict"),
            "passed": old.get("honest_verdict") == "complete_disqualified_required_validation",
        }
    )
    checks.append(
        {
            "check": "original_schema",
            "upstream": "experiment_7709",
            "path": str(old_path),
            "field": "schema",
            "operator": "==",
            "expected": "carnot.exp7709.v671.arc_first_contact.v1",
            "observed": old.get("schema"),
            "passed": old.get("schema") == "carnot.exp7709.v671.arc_first_contact.v1",
        }
    )
    for relative, expected in (
        old.get("source_artifact_hashes", {}).get("pre_gate_receipts", {}).items()
    ):
        path = root / relative
        check = authenticate_bytes(path, expected, "experiment_7708", "pre_gate_sha256")
        checks.append(check)
        if check["passed"]:
            hashes["pre_gate_receipts"][relative] = check["observed"]
        else:
            hashes["missing_custody"].append(str(path))
    rows = rows_doc.get("rows", [])
    checks.append(
        {
            "check": "historical_schema",
            "upstream": "experiment_7709",
            "path": str(raw / "episode_rows.json"),
            "field": "rows.game",
            "operator": "==",
            "expected": ["wa30", "lf52"],
            "observed": [row.get("game") for row in rows],
            "passed": [row.get("game") for row in rows] == ["wa30", "lf52"],
        }
    )
    if not checks[-1]["passed"]:
        return {
            "failed_checks": [row for row in checks if not row["passed"]],
            "checks": checks,
            "hashes": hashes,
            "rows": [],
            "registry_precheck": {},
        }
    checks.append(
        {
            "check": "original_rows_match",
            "upstream": "experiment_7709",
            "path": str(raw / "episode_rows.json"),
            "field": "rows",
            "operator": "==",
            "expected": "immutable_v671_artifact_rows",
            "observed": "immutable_v671_artifact_rows" if rows == old.get("rows") else "mismatch",
            "passed": rows == old.get("rows"),
        }
    )
    joined = join_episode_rows(rows)
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    flat = [(row["episode_id"], action) for row in rows for action in row["actions"]]
    ledger_ok = len(ledger) == len(flat) and all(
        record.get("episode_id") == episode
        and all(
            record.get(key) == action.get(key)
            for key in ("action_index", "action", "observation_id", "level_before", "level_after")
        )
        and record.get("observation", {}).get("observation_id") == action.get("observation_id")
        for record, (episode, action) in zip(ledger, flat, strict=True)
    )
    checks.append(
        {
            "check": "action_ledger_join",
            "upstream": "experiment_7709",
            "path": str(ledger_path),
            "field": "joined_actions",
            "operator": "==",
            "expected": joined["joined_actions"],
            "observed": len(ledger) if ledger_ok else "mismatch",
            "passed": ledger_ok,
        }
    )
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    precheck = {
        item["game"]: {
            "levels_reproduced": item.get("levels_reproduced"),
            "reproduced_levels": list(range(1, int(item.get("levels_reproduced") or 0) + 1)),
            "full_game_clear": item.get("full_game_clear"),
        }
        for item in registry.get("games", [])
        if isinstance(item, dict) and item.get("game") in ("wa30", "lf52")
    }
    reduced_rows: list[Json] = []
    receipts: list[Json] = []
    call_rows: list[Json] = []
    for row in rows:
        game = str(row["game"])
        diagnoses = []
        for request in row["requests"]:
            call = int(request["call_index"])
            for kind in ("request", "response"):
                path = raw / f"{game}__live/requests/{call:02d}_{kind}.json"
                expected = request.get(f"{kind}_sha256")
                check = authenticate_bytes(path, expected, "experiment_7709", f"{kind}_sha256")
                checks.append(check)
                if check["passed"]:
                    hashes["flagged_historical_evidence"][str(path)] = check["observed"]
                else:
                    hashes["missing_custody"].append(str(path))
            if all(check["passed"] for check in checks[-2:]):
                response = json.loads(path.read_text(encoding="utf-8"))
                diagnosis = classify_response(response)
                diagnoses.append(diagnosis)
                call_rows.append(
                    {
                        "game": game,
                        "call_index": call,
                        "response_sha256": request["response_sha256"],
                        **diagnosis,
                    }
                )
        seam_path = raw / f"episodes/{game}/seam_events.jsonl"
        seam_hash = sha256_file(seam_path) if seam_path.is_file() else None
        checks.append(
            {
                "check": "supervisor_receipt_bytes",
                "upstream": "experiment_7709",
                "path": str(seam_path),
                "field": "seam_events_sha256",
                "operator": "!=",
                "expected": None,
                "observed": seam_hash,
                "passed": seam_hash is not None,
            }
        )
        if seam_hash is not None:
            hashes["flagged_historical_evidence"][str(seam_path)] = seam_hash
            receipts.append(_supervisor_receipt(seam_path, game, row["actions"]))
        else:
            hashes["missing_custody"].append(str(seam_path))
        reduced_rows.append(
            {
                "game": game,
                "arm": row["arm"],
                "actions": len(row["actions"]),
                "observations": len(row["observations"]),
                "model_calls": len(row["requests"]),
                "accepted_engines": sum(
                    a.get("accepted") is True for a in row["induction_attempts"]
                ),
                "rejection_reasons": [
                    str(a.get("refinement_rounds", [{}])[0].get("message"))
                    for a in row["induction_attempts"]
                    if a.get("accepted") is not True
                ],
                "observed_level_ups": sum(
                    int(a["level_after"]) > int(a["level_before"]) for a in row["actions"]
                ),
                "observed_level_progress_rate": sum(
                    int(a["level_after"]) > int(a["level_before"]) for a in row["actions"]
                )
                / len(row["actions"]),
                "peak_level": row["peak_level"],
                "censoring": row["censoring"],
                "goal_recall": "unknown",
                "solve_provenance": "live_agent_self_discovery",
                "new_solve_credit": False,
                "response_diagnoses": diagnoses,
                "source_episode_id": row["episode_id"],
                "exclusions": row["exclusions"],
            }
        )
    return {
        "checks": checks,
        "failed_checks": [check for check in checks if not check["passed"]],
        "hashes": hashes,
        "rows": reduced_rows,
        "registry_precheck": precheck,
        "historical_model_provenance": {
            "model": "unsloth/Qwen3.8-27B-GGUF",
            "gguf_sha256": old.get("source_artifact_hashes", {})
            .get("producer_files", {})
            .get("model_weights", {})
            .get("sha256"),
            "calls": len(call_rows),
            "output_tokens": sum(row["completion_tokens"] for row in call_rows),
            "call_rows": call_rows,
        },
        "supervisor": summarize_supervisor(receipts),
        "joined_actions": joined["joined_actions"],
        "prior_verdict": old["honest_verdict"],
    }
