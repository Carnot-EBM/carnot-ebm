"""Discover terminal ARC receipts without launching or changing a game.

The router treats each receipt as evidence, not as a result to trust by name.
It validates the content, live path, and generator before an effect row can
qualify. This prevents a stale path or development run from becoming live
evidence only because it is the newest file on disk.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import yaml


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-agi/spec.md")
OUTPUT_PATH = Path("results/experiment_6857_dynamic_live_arc_receipt_router.json")
# REQ-ARC-WMTE-6642: the eval-run fields this module requires. Checked by
# scripts/eval_run_consumer_field_lint.py against real artifacts + producer source.
EVAL_RUN_FIELDS_READ = (
    "experiment",
    "policy",
    "per_game",
    "honest_verdict",
    "generator_provenance",
    "completions_consumed",
)
V599_PATH = Path("results/experiment_6848_v599_method_change_evidence_contract.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
FLAG_LEDGER_PATH = Path("ops/arc_flag_ledger.yaml")
GENERATOR_SOURCE_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
INFERENCE_SUBSTRATE = "read_only_terminal_live_artifact_discovery"
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

NATIVE_SCHEMA_BY_FAMILY = {
    "supervisor": "carnot.arc.supervisor_receipt.v1",
    "lever_harness": "carnot.arc.lever_harness_receipt.v1",
    "tool_loop": "carnot.arc.tool_loop_receipt.v1",
    "shadow": "carnot.arc.shadow_receipt.v1",
    "canonical_agent": "carnot.arc.canonical_agent_receipt.v1",
}
LEGACY_SCHEMA_FAMILIES = {
    "carnot.experiment_6844.supervisor_action_outcome_credit_audit.v1": "supervisor",
    "carnot.scored_path_lever_ab.v1": "lever_harness",
    "carnot.experiment_6845.tool_gap_causal_support_audit.v1": "tool_loop",
    "carnot.experiment_6846.typed_arc_shadow_monitor.v1": "shadow",
    "carnot.experiment_6776.arc_shadow_supervisor_accrual.v1": "shadow",
}

DEFAULT_DISCOVERY_ROOTS = [
    {"path": "results/arc_leaderboard_eval_runs", "patterns": ["*.json"]},
    {
        "path": "results",
        "patterns": ["*supervisor*.json", "*lever*.json", "*tool_gap*.json", "*shadow*.json"],
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "discovery_roots",
    "accepted_schema_manifest",
    "rejected_source_manifest",
    "rows",
    "provenance_qualified_manifest",
    "generator_provenance_rows",
    "live_reachability_rows",
    "configuration_strata",
    "supervisor_headroom_rows",
    "first_party_tool_gap_rows",
    "unmatched_receipt_rows",
    "arc_receipt_router_complete_score",
    "supervisor_headroom_ready_score",
    "tool_gap_first_party_receipts_ready_score",
    "solve_claim",
    "game_level_solve_count",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each field states why it exists, so later consumers keep the evidence boundary.",
    "preconditions_checked": "The router must fail before evidence credit when an owned input is unavailable.",
    "inference_substrate": "The declaration proves that discovery reads terminal files and launches no game.",
    "duration_s": "Measured wall time helps detect a fabricated or stalled discovery pass.",
    "source_artifact_hashes": "Hashes freeze the exact bytes that the manifest describes.",
    "discovery_roots": "Declared roots make the execution-time search reproducible without a pinned experiment path.",
    "accepted_schema_manifest": "Explicit schemas prevent a filename or modification time from creating authority.",
    "rejected_source_manifest": "Rejected inputs remain visible, so exclusion cannot hide inconvenient evidence.",
    "rows": "One normalized row per receipt keeps actions, tools, and outcomes independently auditable.",
    "provenance_qualified_manifest": "Only sources with current generators and a live seam can support effects.",
    "generator_provenance_rows": "Generator identity prevents retired or changed models from entering live evidence.",
    "live_reachability_rows": "Reachability separates the submitted agent path from development proxies.",
    "configuration_strata": "Separate strata prevent policy, budget, model, or mode changes from being pooled.",
    "supervisor_headroom_rows": "Headroom is computed before effect, so a zero-opportunity null is not evidence.",
    "first_party_tool_gap_rows": "First-party tool chains stay separate from transport-only or proxy receipts.",
    "unmatched_receipt_rows": "Missing exact links stay diagnostic and cannot silently become causal credit.",
    "arc_receipt_router_complete_score": "Downstream work needs a stable signal that discovery itself completed.",
    "supervisor_headroom_ready_score": "Supervisor effect work requires at least one matched nonzero-headroom row.",
    "tool_gap_first_party_receipts_ready_score": "Tool readiness requires an agent-visible first-party chain.",
    "solve_claim": "Discovery does not run a game and therefore cannot claim a solve.",
    "game_level_solve_count": "A zero count prevents receipt routing from changing the solve registry by implication.",
    "gate_check_summary": "Every blocked result names the failed check and exact observed value.",
    "verifier_is_oracle": "External receipts define outcomes; this router is not the correctness oracle.",
    "verdict_class": "A closed class lets automated consumers interpret the terminal result safely.",
    "honest_verdict": "A complete_ prefix marks the router run as terminal even when a gate blocks it.",
}


def accepted_schema_manifest() -> list[JsonDict]:
    """Return the content contracts used for selection and legacy intake."""

    common = [
        "experiment_id",
        "terminal_status",
        "attempt_identity",
        "generator_provenance",
        "live_reachability",
        "configuration",
        "rows",
        "declared_content_sha256",
    ]
    rows = [
        {
            "schema": schema,
            "artifact_family": family,
            "kind": "native",
            "required_fields": common,
        }
        for family, schema in NATIVE_SCHEMA_BY_FAMILY.items()
    ]
    rows.extend(
        {
            "schema": schema,
            "artifact_family": family,
            "kind": "legacy_read_only",
            "required_fields": ["honest_verdict"],
        }
        for schema, family in LEGACY_SCHEMA_FAMILIES.items()
    )
    rows.append(
        {
            "schema": "arc_leaderboard_eval.content.v1",
            "artifact_family": "canonical_agent",
            "kind": "legacy_read_only",
            "required_fields": [
                "experiment",
                "policy",
                "budget",
                "random_seed",
                "per_game",
                "honest_verdict",
            ],
        }
    )
    return rows


def canonical_bytes(value: Any) -> bytes:
    """Encode JSON deterministically so hashes do not depend on formatting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return the repository's labeled SHA-256 form."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash file bytes exactly as they exist at discovery time."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while excluding its self-referential declared hash."""

    copy = deepcopy(dict(payload))
    copy.pop("declared_content_sha256", None)
    return sha256_bytes(canonical_bytes(copy))


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable output fields while excluding runtime duration and the hash itself."""

    copy = deepcopy(dict(artifact))
    copy.pop("duration_s", None)
    copy.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_bytes(copy))


def valid_run_date(value: str) -> bool:
    """Accept the conductor's fixed YYYYMMDD execution-date form."""

    try:
        return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        return False


def parse_discovery_root(value: str) -> JsonDict:
    """Parse PATH:PATTERN,PATTERN without treating a missing pattern as broad access."""

    path, separator, raw_patterns = value.partition(":")
    if not path:
        raise ValueError("empty discovery root")
    patterns = [item for item in raw_patterns.split(",") if item] if separator else ["*.json"]
    return {"path": path, "patterns": patterns or ["*.json"]}


def discover_candidates(
    repo_root: Path, roots: Sequence[Mapping[str, Any]] | None = None
) -> list[JsonDict]:
    """Read candidate JSON objects from declared roots in stable path order."""

    resolved_root = repo_root.resolve()
    output: list[JsonDict] = []
    seen: set[Path] = set()
    for root_spec in roots or DEFAULT_DISCOVERY_ROOTS:
        relative_root = Path(str(root_spec["path"]))
        directory = (resolved_root / relative_root).resolve()
        try:
            directory.relative_to(resolved_root)
        except ValueError:
            output.append(
                {
                    "path": relative_root.as_posix(),
                    "exists": False,
                    "load_error": "discovery_root_outside_repo",
                }
            )
            continue
        if not directory.is_dir():
            output.append(
                {
                    "path": relative_root.as_posix(),
                    "exists": False,
                    "load_error": "discovery_root_missing",
                }
            )
            continue
        paths = sorted(
            {
                path.resolve()
                for pattern in root_spec.get("patterns", ["*.json"])
                for path in directory.glob(str(pattern))
                if path.is_file()
            }
        )
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            relative = path.relative_to(resolved_root).as_posix()
            record: JsonDict = {
                "path": relative,
                "exists": True,
                "file_sha256": sha256_file(path),
                "load_error": None,
            }
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("top-level JSON must be an object")
                record["payload"] = payload
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
                record["load_error"] = f"{type(exc).__name__}:{exc}"
            output.append(record)
    return output


def _schema_family(payload: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Identify a family from content only, without consulting the filename."""

    schema = str(payload.get("schema") or "")
    native = {value: key for key, value in NATIVE_SCHEMA_BY_FAMILY.items()}
    if schema in native:
        return native[schema], schema, "native"
    if schema in LEGACY_SCHEMA_FAMILIES:
        return LEGACY_SCHEMA_FAMILIES[schema], schema, "legacy_read_only"
    if payload.get("experiment") in {"arc_leaderboard_eval", "arc_live_oracle_gap"}:
        return "canonical_agent", "arc_leaderboard_eval.content.v1", "legacy_read_only"
    return None, schema or None, None


def _terminal(payload: Mapping[str, Any], kind: str) -> bool:
    """Recognize complete terminal content while excluding partial work."""

    status = str(
        payload.get("terminal_status") if kind == "native" else payload.get("status") or ""
    )
    verdict = str(payload.get("honest_verdict") or "")
    verdict_class = str(payload.get("verdict_class") or "")
    if "partial" in status.lower() or "partial" in verdict.lower() or verdict_class == "partial":
        return False
    if kind == "native":
        return status == "complete" and verdict.startswith("complete_")
    return verdict.startswith(("complete_", "complete:")) or status.startswith("complete")


def _current_generator_model(repo_root: Path) -> str | None:
    """Read the canonical model constant without importing the large agent module."""

    path = repo_root / GENERATOR_SOURCE_PATH
    if not path.is_file():
        return None
    marker = 'ARC_LIVE_GENERATOR_REPO_SUBSTR = "'
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(marker) and line.endswith('"'):
            return line[len(marker) : -1]
    return None


def _legacy_generator(payload: Mapping[str, Any], family: str, repo_root: Path) -> JsonDict | None:
    """Convert only generator receipts that bind each canonical-agent row to the current model."""

    if family != "canonical_agent" or payload.get("policy") != "e3":
        return None
    rows = payload.get("per_game")
    if not isinstance(rows, list) or not rows:
        return None
    expected = _current_generator_model(repo_root)
    observed: list[str] = []
    for row in rows:
        provenance = row.get("generator_provenance") if isinstance(row, dict) else None
        completions = row.get("completions_consumed") if isinstance(row, dict) else None
        path = (
            provenance.get("observed_server_model_path") if isinstance(provenance, dict) else None
        )
        used = completions.get("completions", 0) if isinstance(completions, dict) else 0
        if (
            not expected
            or not isinstance(path, str)
            or expected.lower() not in path.lower()
            or provenance.get("resolved") is not True
            or provenance.get("reuse_model_check") != "match"
            or int(used or 0) <= 0
        ):
            return None
        observed.append(path)
    implementation = repo_root / GENERATOR_SOURCE_PATH
    return {
        "generator_id": expected,
        "implementation_path": GENERATOR_SOURCE_PATH.as_posix(),
        "implementation_sha256": sha256_file(implementation),
        "configuration_sha256": sha256_bytes(canonical_bytes(observed)),
        "model_id": expected,
        "observed_model_paths": observed,
        "validation": "runtime_model_match_and_completion_receipt",
    }


def _legacy_reachability(payload: Mapping[str, Any], family: str) -> JsonDict:
    """State reachability conservatively for historical schemas."""

    return {
        "reachable": family == "canonical_agent" and payload.get("policy") == "e3",
        "entrypoint": "make_carnot_agent -> E3AgentPolicy",
        "live_seam": "canonical_action_seam",
        "development_proxy": payload.get("solve_provenance") == "development_proxy",
        "outer_loop_re": payload.get("solve_provenance") == "outer_loop_re",
        "source_reading": bool(payload.get("read_game_source") or payload.get("used_env_source")),
    }


def _native_rejection(payload: Mapping[str, Any], repo_root: Path) -> str | None:
    """Return the first provenance failure in scientific-priority order."""

    required = {
        "experiment_id",
        "terminal_status",
        "attempt_identity",
        "generator_provenance",
        "live_reachability",
        "configuration",
        "rows",
        "declared_content_sha256",
    }
    if required - set(payload):
        return "partial_artifact_missing_fields"
    provenance = payload.get("generator_provenance")
    if not isinstance(provenance, dict):
        return "missing_generator_provenance"
    implementation_path = provenance.get("implementation_path")
    declared_implementation = provenance.get("implementation_sha256")
    if not implementation_path or not declared_implementation:
        return "missing_generator_provenance"
    implementation = (repo_root / str(implementation_path)).resolve()
    try:
        implementation.relative_to(repo_root.resolve())
    except ValueError:
        return "changed_generator"
    if not implementation.is_file() or sha256_file(implementation) != declared_implementation:
        return "changed_generator"
    reachability = payload.get("live_reachability")
    if not isinstance(reachability, dict):
        return "live_seam_unreachable"
    for field, reason in (
        ("development_proxy", "development_proxy"),
        ("outer_loop_re", "outer_loop_re"),
        ("source_reading", "source_reading"),
    ):
        if reachability.get(field) is True:
            return reason
    if reachability.get("reachable") is not True:
        return "live_seam_unreachable"
    if payload.get("pooled_policy_change") is True:
        return "pooled_policy_change"
    if payload.get("verification_failure") is True or payload.get("flagged_adversarial") is True:
        return "flagged_verification_failure"
    configuration = payload.get("configuration")
    policy_hash = configuration.get("policy_hash") if isinstance(configuration, dict) else None
    if not policy_hash:
        return "mixed_policy_configuration"
    if any(
        isinstance(row, dict) and row.get("policy_hash") not in (None, policy_hash)
        for row in payload.get("rows", [])
    ):
        return "mixed_policy_configuration"
    if payload_sha256(payload) != payload.get("declared_content_sha256"):
        return "stale_declared_hash"
    return None


def _candidate_metadata(
    candidate: Mapping[str, Any], family: str, schema: str, kind: str
) -> JsonDict:
    """Project source metadata without carrying the full payload into the result."""

    payload = candidate["payload"]
    return {
        "source_path": candidate["path"],
        "source_sha256": candidate["file_sha256"],
        "schema": schema,
        "schema_kind": kind,
        "artifact_family": family,
        "experiment_id": payload.get("experiment_id", payload.get("experiment")),
        "terminal_status": payload.get("terminal_status", payload.get("status", "complete")),
        "attempt_identity": _artifact_attempt(payload, family, candidate["file_sha256"]),
        "stored_source_path_ignored": bool(
            payload.get("stored_source_path") or payload.get("stored_absolute_path")
        ),
    }


def _artifact_attempt(payload: Mapping[str, Any], family: str, source_hash: str) -> str:
    """Use declared attempt identity, or derive one from exact legacy run fields."""

    declared = payload.get("attempt_identity")
    if declared:
        return str(declared)
    identity = {
        "family": family,
        "experiment": payload.get(
            "experiment_id", payload.get("experiment", payload.get("schema"))
        ),
        "run_date": payload.get("run_date"),
        "seed": payload.get("random_seed"),
        "policy": payload.get("policy"),
        "budget": payload.get("budget"),
        "source_hash": source_hash if family != "canonical_agent" else None,
    }
    return sha256_bytes(canonical_bytes(identity))


def _legacy_configuration(
    payload: Mapping[str, Any], row: Mapping[str, Any] | None = None
) -> JsonDict:
    """Build a full configuration record and state every absent mode as unknown."""

    row = row or {}
    supervisor = row.get("trajectory_supervisor")
    supervisor_mode = (
        supervisor.get("mode", "on")
        if isinstance(supervisor, dict) and supervisor.get("enabled")
        else "off"
    )
    tool_mode = "on" if row.get("tool_loop_enabled") is True else "off_or_unobserved"
    basis = {
        "policy": payload.get("policy", row.get("arm", "unknown")),
        "budget": payload.get("budget", row.get("budget")),
        "supervisor_mode": supervisor_mode,
        "tool_mode": tool_mode,
    }
    return {
        "game": row.get("game", "unknown"),
        "policy_hash": sha256_bytes(canonical_bytes(basis)),
        "budget": basis["budget"],
        "supervisor_mode": supervisor_mode,
        "tool_mode": tool_mode,
    }


def _normalize_native(candidate: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[JsonDict]:
    """Attach source and configuration provenance to each native receipt row."""

    payload = candidate["payload"]
    configuration = payload["configuration"]
    provenance = payload["generator_provenance"]
    reachability = payload["live_reachability"]
    output: list[JsonDict] = []
    for source_row in payload["rows"]:
        row = deepcopy(source_row)
        row.update(
            {
                "source_path": metadata["source_path"],
                "source_sha256": metadata["source_sha256"],
                "experiment_id": metadata["experiment_id"],
                "terminal_status": metadata["terminal_status"],
                "artifact_family": metadata["artifact_family"],
                "game": configuration.get("game"),
                "model": provenance.get("model_id"),
                "generator_provenance": deepcopy(provenance),
                "policy_hash": configuration.get("policy_hash"),
                "budget": configuration.get("budget"),
                "supervisor_mode": configuration.get("supervisor_mode"),
                "tool_mode": configuration.get("tool_mode"),
                "attempt_identity": source_row.get(
                    "attempt_identity", metadata["attempt_identity"]
                ),
                "live_seam": reachability.get("live_seam"),
            }
        )
        output.append(row)
    return output


def _normalize_legacy_canonical(
    candidate: Mapping[str, Any], metadata: Mapping[str, Any], provenance: Mapping[str, Any]
) -> list[JsonDict]:
    """Turn each leaderboard game into a source row and one exact level outcome."""

    payload = candidate["payload"]
    output: list[JsonDict] = []
    for game_row in payload.get("per_game", []):
        configuration = _legacy_configuration(payload, game_row)
        attempt = sha256_bytes(
            canonical_bytes(
                {
                    "artifact_attempt": metadata["attempt_identity"],
                    "game": game_row.get("game"),
                    "configuration": configuration,
                }
            )
        )
        common = {
            "source_path": metadata["source_path"],
            "source_sha256": metadata["source_sha256"],
            "experiment_id": metadata["experiment_id"],
            "terminal_status": metadata["terminal_status"],
            "artifact_family": "canonical_agent",
            "game": configuration["game"],
            "model": provenance.get("model_id"),
            "generator_provenance": deepcopy(dict(provenance)),
            "policy_hash": configuration["policy_hash"],
            "budget": configuration["budget"],
            "supervisor_mode": configuration["supervisor_mode"],
            "tool_mode": configuration["tool_mode"],
            "attempt_identity": attempt,
            "live_seam": "canonical_action_seam",
        }
        output.append(
            {
                **common,
                "row_kind": "source",
                "row_identity": f"{attempt}:source",
                "actions": game_row.get("actions"),
            }
        )
        output.append(
            {
                **common,
                "row_kind": "exact_outcome",
                "row_identity": f"{attempt}:outcome",
                "levels_before": 0,
                "levels_after": game_row.get("levels"),
                "level_ceiling": game_row.get("oracle_levels", game_row.get("levels")),
                "exact": True,
            }
        )
    return output


def _reject(path: str, source_hash: Any, schema: Any, family: Any, reason: str) -> JsonDict:
    """Create one stable rejected-source record."""

    return {
        "source_path": path,
        "source_sha256": source_hash,
        "schema": schema,
        "artifact_family": family,
        "reason": reason,
    }


def route_candidates(repo_root: Path, candidates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Qualify sources, quarantine ambiguity, and normalize exact receipt rows."""

    rejected: list[JsonDict] = []
    staged: list[JsonDict] = []
    for candidate in candidates:
        if candidate.get("load_error"):
            rejected.append(
                _reject(
                    str(candidate.get("path")),
                    candidate.get("file_sha256"),
                    None,
                    None,
                    str(candidate["load_error"]),
                )
            )
            continue
        payload = candidate["payload"]
        family, schema, kind = _schema_family(payload)
        if not family or not schema or not kind:
            rejected.append(
                _reject(
                    candidate["path"],
                    candidate.get("file_sha256"),
                    schema,
                    None,
                    "schema_not_accepted",
                )
            )
            continue
        if not _terminal(payload, kind):
            rejected.append(
                _reject(
                    candidate["path"],
                    candidate.get("file_sha256"),
                    schema,
                    family,
                    "nonterminal_artifact",
                )
            )
            continue
        metadata = _candidate_metadata(candidate, family, schema, kind)
        if kind == "native":
            reason = _native_rejection(payload, repo_root)
            provenance = payload.get("generator_provenance")
            reachability = payload.get("live_reachability")
        else:
            reachability = _legacy_reachability(payload, family)
            provenance = _legacy_generator(payload, family, repo_root)
            reason = None
            if reachability.get("development_proxy"):
                reason = "development_proxy"
            elif reachability.get("outer_loop_re"):
                reason = "outer_loop_re"
            elif reachability.get("source_reading"):
                reason = "source_reading"
            elif (
                payload.get("flagged_adversarial") is True
                or payload.get("verification_failure") is True
            ):
                reason = "flagged_verification_failure"
            elif provenance is None:
                reason = "missing_generator_provenance"
        if reason:
            rejected.append(
                _reject(candidate["path"], candidate.get("file_sha256"), schema, family, reason)
            )
            continue
        staged.append(
            {
                "candidate": candidate,
                "metadata": metadata,
                "provenance": provenance,
                "reachability": reachability,
            }
        )

    groups: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for item in staged:
        metadata = item["metadata"]
        groups[(metadata["artifact_family"], metadata["attempt_identity"])].append(item)
    qualified: list[JsonDict] = []
    for items in groups.values():
        if len({item["metadata"]["source_sha256"] for item in items}) > 1:
            for item in items:
                metadata = item["metadata"]
                rejected.append(
                    _reject(
                        metadata["source_path"],
                        metadata["source_sha256"],
                        metadata["schema"],
                        metadata["artifact_family"],
                        "ambiguous_attempt_identity",
                    )
                )
        else:
            qualified.extend(items)

    manifest: list[JsonDict] = []
    rows: list[JsonDict] = []
    generator_rows: list[JsonDict] = []
    reachability_rows: list[JsonDict] = []
    for item in sorted(qualified, key=lambda value: value["metadata"]["source_path"]):
        metadata = item["metadata"]
        manifest.append(dict(metadata))
        generator_rows.append(
            {
                "source_path": metadata["source_path"],
                "attempt_identity": metadata["attempt_identity"],
                **deepcopy(dict(item["provenance"])),
            }
        )
        reachability_rows.append(
            {
                "source_path": metadata["source_path"],
                "attempt_identity": metadata["attempt_identity"],
                **deepcopy(dict(item["reachability"])),
            }
        )
        if metadata["schema_kind"] == "native":
            rows.extend(_normalize_native(item["candidate"], metadata))
        else:
            rows.extend(
                _normalize_legacy_canonical(item["candidate"], metadata, item["provenance"])
            )
    return {
        "rejected": sorted(rejected, key=lambda row: (row["source_path"], row["reason"])),
        "manifest": manifest,
        "rows": rows,
        "generator_rows": generator_rows,
        "reachability_rows": reachability_rows,
    }


def _same_attempt(row: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    """Require one source and attempt for every exact join."""

    return row.get("source_path") == other.get("source_path") and row.get(
        "attempt_identity"
    ) == other.get("attempt_identity")


def join_receipts(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Join supervisor and tool links to exact same-attempt receipts."""

    by_identity: dict[str, Mapping[str, Any]] = {
        str(row.get("row_identity")): row for row in rows if row.get("row_identity")
    }
    supervisor: list[JsonDict] = []
    tools: list[JsonDict] = []
    unmatched: list[JsonDict] = []
    for row in rows:
        kind = row.get("row_kind")
        if kind == "supervisor_action":
            next_action = by_identity.get(str(row.get("next_action_identity")))
            transition = by_identity.get(str(row.get("transition_identity")))
            outcome = by_identity.get(str(row.get("later_outcome_identity")))
            if (
                not outcome
                or outcome.get("row_kind") != "exact_outcome"
                or outcome.get("exact") is not True
            ):
                unmatched.append({**dict(row), "reason": "missing_exact_later_outcome"})
                continue
            if (
                not next_action
                or not transition
                or not all(_same_attempt(row, item) for item in (next_action, transition, outcome))
            ):
                unmatched.append({**dict(row), "reason": "missing_exact_action_transition_join"})
                continue
            if not row.get("matched_opportunity_identity"):
                unmatched.append({**dict(row), "reason": "missing_matched_opportunity"})
                continue
            before = outcome.get("levels_before")
            ceiling = outcome.get("level_ceiling")
            if not isinstance(before, (int, float)) or not isinstance(ceiling, (int, float)):
                unmatched.append({**dict(row), "reason": "missing_exact_headroom_fields"})
                continue
            supervisor.append(
                {
                    **dict(row),
                    "next_action": dict(next_action),
                    "transition": dict(transition),
                    "later_exact_outcome": dict(outcome),
                    "exact_outcome_headroom": max(0, ceiling - before),
                }
            )
        elif kind == "tool_event":
            if not row.get("agent_visible_receipt"):
                unmatched.append({**dict(row), "reason": "missing_agent_visible_receipt"})
                continue
            next_action = by_identity.get(str(row.get("next_action_identity")))
            outcome = by_identity.get(str(row.get("later_outcome_identity")))
            if (
                not outcome
                or outcome.get("row_kind") != "exact_outcome"
                or outcome.get("exact") is not True
            ):
                unmatched.append({**dict(row), "reason": "missing_exact_later_outcome"})
                continue
            if not next_action or not all(
                _same_attempt(row, item) for item in (next_action, outcome)
            ):
                unmatched.append({**dict(row), "reason": "missing_exact_next_action_join"})
                continue
            if row.get("first_party") is True:
                tools.append(
                    {
                        **dict(row),
                        "next_action": dict(next_action),
                        "later_exact_outcome": dict(outcome),
                    }
                )
            else:
                unmatched.append({**dict(row), "reason": "not_first_party_tool_chain"})
    return supervisor, tools, unmatched


def configuration_strata(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Group only byte-equal mechanism configurations."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        identity_fields = {
            "artifact_family": row.get("artifact_family"),
            "game": row.get("game"),
            "model": row.get("model"),
            "policy_hash": row.get("policy_hash"),
            "budget": row.get("budget"),
            "supervisor_mode": row.get("supervisor_mode"),
            "tool_mode": row.get("tool_mode"),
        }
        groups[sha256_bytes(canonical_bytes(identity_fields))].append(row)
    output: list[JsonDict] = []
    for identity, grouped in sorted(groups.items()):
        first = grouped[0]
        output.append(
            {
                "stratum_identity": identity,
                "artifact_family": first.get("artifact_family"),
                "game": first.get("game"),
                "model": first.get("model"),
                "policy_hash": first.get("policy_hash"),
                "budget": first.get("budget"),
                "supervisor_mode": first.get("supervisor_mode"),
                "tool_mode": first.get("tool_mode"),
                "row_count": len(grouped),
            }
        )
    return output


def _read_json(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one owned JSON precondition with a named error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (value, None) if isinstance(value, dict) else (None, "top_level_not_object")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _read_yaml(path: Path) -> tuple[JsonDict | None, str | None]:
    """Read one owned YAML precondition without accepting a scalar document."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return (value, None) if isinstance(value, dict) else (None, "top_level_not_mapping")
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Keep the expected and observed values beside every gate decision."""

    return {"check": check, "expected": expected, "observed": observed, "passed": passed}


def precondition_checks(repo_root: Path) -> list[JsonDict]:
    """Evaluate owned inputs and the deliberately empty solve scope."""

    contract, contract_error = _read_json(repo_root / V599_PATH)
    registry, registry_error = _read_yaml(repo_root / REGISTRY_PATH)
    flags, flags_error = _read_yaml(repo_root / FLAG_LEDGER_PATH)
    ready = contract.get("v599_evidence_contract_ready_score") if contract else contract_error
    return [
        _check("v599_evidence_contract_ready_score", 1, ready, ready == 1),
        _check("arc_registry_readable", True, registry_error or True, registry is not None),
        _check("arc_flag_ledger_readable", True, flags_error or True, flags is not None),
        _check("game_source_files_read", [], [], True),
        _check("duplicate_solve_scope", [], [], True),
    ]


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first failure for consumers and retain every check for audit."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_check": failed[0]["check"] if failed else None,
        "observed": failed[0]["observed"] if failed else "all checks pass",
        "failed_checks": failed,
        "checks": [dict(row) for row in checks],
    }


def sample_live_processes() -> list[JsonDict]:
    """Take one read-only process snapshot and never signal or poll a process."""

    command = ["ps", "-eo", "pid=,ppid=,lstart=,stat=,etime=,args=", "--sort=pid"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"observation_error": f"{type(exc).__name__}:{exc}"}]
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = line.split(None, 8)
        if len(parts) != 9 or "arc" not in parts[8].lower():
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "process_start": " ".join(parts[2:7]),
                "state": parts[7],
                "elapsed": parts[8].split(None, 1)[0],
                "command_sha256": sha256_bytes(parts[8].encode()),
                "observation_only": True,
            }
        )
    return rows


def _source_hash_manifest(
    repo_root: Path, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Freeze owned preconditions and every discovered candidate by relative path."""

    manifest: dict[str, JsonDict] = {}
    for path in (V599_PATH, REGISTRY_PATH, FLAG_LEDGER_PATH):
        full = repo_root / path
        manifest[path.as_posix()] = {
            "path": path.as_posix(),
            "exists": full.is_file(),
            "file_sha256": sha256_file(full) if full.is_file() else None,
        }
    for candidate in candidates:
        path = str(candidate.get("path"))
        manifest[path] = {
            "path": path,
            "exists": candidate.get("exists", False),
            "file_sha256": candidate.get("file_sha256"),
            "load_error": candidate.get("load_error"),
        }
    return manifest


def build_artifact(
    repo_root: Path,
    *,
    run_date: str,
    duration_s: float,
    discovery_roots: Sequence[Mapping[str, Any]] | None = None,
    process_observations: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the complete terminal manifest in memory without changing source evidence."""

    roots = [deepcopy(dict(row)) for row in (discovery_roots or DEFAULT_DISCOVERY_ROOTS)]
    candidates = discover_candidates(repo_root, roots)
    routed = route_candidates(repo_root, candidates)
    rows = routed["rows"]
    supervisor, tools, unmatched = join_receipts(rows)
    identities = [str(row.get("row_identity")) for row in rows if row.get("row_identity")]
    duplicates = sorted(identity for identity, count in Counter(identities).items() if count > 1)
    checks = precondition_checks(repo_root)
    checks.append(_check("duplicate_row_identity", [], duplicates, not duplicates))
    complete = int(all(row["passed"] for row in checks))
    summary = _gate_summary(checks)
    artifact: JsonDict = {
        "schema": "carnot.experiment_6857.dynamic_live_arc_receipt_router.v1",
        "experiment_id": "exp6857-dynamic-live-arc-receipt-router",
        "run_date": run_date,
        "status": "complete" if complete else "complete_blocked",
        "field_principles": {},
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": _source_hash_manifest(repo_root, candidates),
        "discovery_roots": roots,
        "accepted_schema_manifest": accepted_schema_manifest(),
        "rejected_source_manifest": routed["rejected"],
        "rows": rows,
        "provenance_qualified_manifest": routed["manifest"],
        "generator_provenance_rows": routed["generator_rows"],
        "live_reachability_rows": routed["reachability_rows"],
        "configuration_strata": configuration_strata(rows),
        "supervisor_headroom_rows": supervisor,
        "first_party_tool_gap_rows": tools,
        "unmatched_receipt_rows": unmatched,
        "process_observations": [dict(row) for row in (process_observations or [])],
        "arc_receipt_router_complete_score": complete,
        "supervisor_headroom_ready_score": int(
            any(row["exact_outcome_headroom"] > 0 for row in supervisor)
        ),
        "tool_gap_first_party_receipts_ready_score": int(bool(tools)),
        "solve_claim": False,
        "game_level_solve_count": 0,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "null" if complete else "blocked",
        "honest_verdict": (
            "complete_dynamic_live_arc_receipt_router"
            if complete
            else "complete_blocked_dynamic_live_arc_receipt_router"
        ),
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"REQ-ARC-6857 requires {key} for auditability.")
        for key in artifact
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate the stable downstream contract without trusting its ready score."""

    errors = [
        f"missing top-level field: {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim must be false")
    if artifact.get("game_level_solve_count") != 0:
        errors.append("game_level_solve_count must be 0")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("invalid verdict_class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid inference_substrate")
    checksum = artifact.get("reproducibility_checksum")
    if checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write through a sibling temporary file so readers never see partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os_getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def os_getpid() -> int:
    """Keep process identity behind a small testable function."""

    import os

    return os.getpid()


def main(argv: Sequence[str] | None = None) -> int:
    """Run one read-only discovery pass and freeze its terminal artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--discovery-root", action="append", default=[])
    args = parser.parse_args(argv)
    if not valid_run_date(args.date):
        parser.error("--date must use YYYYMMDD")
    try:
        roots = (
            [parse_discovery_root(value) for value in args.discovery_root]
            if args.discovery_root
            else DEFAULT_DISCOVERY_ROOTS
        )
    except ValueError as exc:
        parser.error(str(exc))
    started = time.perf_counter()
    artifact = build_artifact(
        args.root,
        run_date=args.date,
        duration_s=0.0,
        discovery_roots=roots,
        process_observations=sample_live_processes(),
    )
    artifact["duration_s"] = round(time.perf_counter() - started, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    output = args.output or args.root / OUTPUT_PATH
    write_json_atomic(output, artifact)
    errors = validate_artifact(artifact)
    if errors:
        print(json.dumps({"validation_errors": errors}, indent=2))
        return 2
    print(json.dumps({"output": str(output), "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper exercises this process boundary
    raise SystemExit(main())
