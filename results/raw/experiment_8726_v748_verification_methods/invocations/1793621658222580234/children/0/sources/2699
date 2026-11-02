"""Rebuild Exp7127 from raw traces without trusting producer projections.

The audit hashes bytes before it parses claims. It joins each executed action
to its proposal, schema check, transition, reward, and level receipt. It also
checks source-level adapter access because deleting one registry key does not
remove code that the worker already imported.

Spec refs: REQ-ARC-7128 and SCENARIO-ARC-7128-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_RELATIVE_PATH = Path("results/experiment_7127_v626_adapter_withheld_arc_loo.json")
RESULT_RELATIVE_PATH = Path("results/experiment_7128_v626_arc_loo_causal_audit.json")
PRODUCER_RELATIVE_PATH = Path("python/carnot/experiment_7127_v626_adapter_withheld_arc_loo.py")
ADAPTER_RELATIVE_PATH = Path("python/carnot/agentic/arc_game_adapters.py")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
DEFAULT_RAW_ROOT = Path("/tmp/carnot-exp7127-v626-adapter-withheld-arc-loo")
ARM_NAMES = ("adapter_withheld", "adapter_visible_control")
REAL_ENTRYPOINT = "make_carnot_agent:E3AgentPolicy"
EXACT_VERIFIER = "E3 policy-visible action schema"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts: independent ARC raw-trace audit"
RANDOM_SEED = 7_128_202_609_08

VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
VERDICT_PREFIXES = {
    "positive": "complete_positive_",
    "circular_positive": "complete_circular_",
    "null": "complete_null_",
    "blocked": "blocked_",
    "disqualified": "complete_disqualified_",
    "partial": "partial_",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "upstream_verdict_class",
    "upstream_honest_verdict",
    "rows",
    "per_game_results",
    "raw_hash_rows",
    "process_isolation_rows",
    "adapter_access_rows",
    "forbidden_read_rows",
    "entrypoint_rows",
    "action_provenance_rows",
    "transition_recompute_rows",
    "removal_replay_rows",
    "causal_credit_rows",
    "level_rows",
    "registry_hash_before",
    "registry_hash_after",
    "registry_mutated",
    "solve_provenance",
    "solve_claim_made",
    "offline_reproduced",
    "arc_causal_audit_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Explains why each audit field exists.",
    "preconditions_checked": "Records source availability before claims are used.",
    "run_date": "Pins the requested audit date.",
    "inference_substrate": "Identifies an independent artifact and trace aggregation.",
    "inference_substrate_class": "Separates an executed audit from a missing-input block.",
    "execution_venue": "Records the host execution boundary.",
    "duration_s": "Reports the audit wall time.",
    "source_artifact_hashes": "Binds every source that the audit inspected.",
    "upstream_verdict_class": "Preserves the Exp7127 terminal class without trusting it.",
    "upstream_honest_verdict": "Preserves the Exp7127 conclusion for comparison.",
    "rows": "Keeps independently rebuilt per-arm metrics visible.",
    "per_game_results": "Keeps the selected game and rebuilt delta visible.",
    "raw_hash_rows": "Checks path, bytes, hashes, JSONL, and event counts per trace.",
    "process_isolation_rows": "Joins raw start, finish, phase, and upstream process identities.",
    "adapter_access_rows": "Detects target recipes retained after a registry-key deletion.",
    "forbidden_read_rows": "Reports recorded policy reads or their explicit unavailability.",
    "entrypoint_rows": "Binds receipts to the real factory and E3 policy source path.",
    "action_provenance_rows": "Joins each proposal to the complete executed receipt chain.",
    "transition_recompute_rows": "Rebuilds arm counts, levels, and phase durations per unit.",
    "removal_replay_rows": "Measures supported removals and labels missing inputs unavailable.",
    "causal_credit_rows": "Prevents labels and non-executed proposals from earning causal credit.",
    "level_rows": "Checks development-only solve provenance for every level receipt.",
    "registry_hash_before": "Preserves the producer's pre-run registry identity.",
    "registry_hash_after": "Records the registry bytes observed by this audit.",
    "registry_mutated": "Shows whether producer or current registry bytes differ.",
    "solve_provenance": "Keeps all findings in development-proxy scope.",
    "solve_claim_made": "Prevents this audit from creating a solve claim.",
    "offline_reproduced": "Prevents an audit from becoming reproduction evidence.",
    "arc_causal_audit_complete_score": "Is one only when every available unit reached a terminal audit result.",
    "random_seed": "Pins deterministic removal replay ordering.",
    "reproducibility_checksum": "Detects changes to the completed artifact.",
    "gate_check_summary": "Names exact expected and observed values for the first failure.",
    "verifier_is_oracle": "States that schema checks do not supply game truth.",
    "verdict_class": "Uses the closed terminal class set.",
    "honest_verdict": "Provides a class-consistent machine-readable conclusion.",
}

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_READ_MARKERS = (
    "arc_game_adapters",
    "arc_solve_registry",
    "registry_traject",
    "known_action",
    "solution",
    "checkpoint",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable bytes for a JSON-compatible value."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest for exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash one file as bytes, or return None when it is unavailable."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after removing its self-reference."""

    projected = deepcopy(dict(artifact))
    projected["reproducibility_checksum"] = ""
    return sha256_bytes(canonical_json_bytes(projected))


def gate_check(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Record both sides of one exact audit check."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Promote the first failed check while retaining all checks."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else None,
        "observed_value": failed.get("observed_value") if failed else None,
        "all_passed": failed is None,
    }


def _empty_evidence() -> JsonDict:
    return {field: [] for field in REQUIRED_ARTIFACT_FIELDS if field.endswith("_rows") or field == "rows"}


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    evidence: Mapping[str, Any],
    verdict_class: str,
    honest_verdict: str,
    inference_substrate_class: str = "aggregation",
) -> JsonDict:
    """Build the fixed artifact schema from independently computed evidence."""

    empty = _empty_evidence()
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(preconditions_checked)),
        "run_date": str(run_date),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": inference_substrate_class,
        "execution_venue": "host",
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "upstream_verdict_class": evidence.get("upstream_verdict_class"),
        "upstream_honest_verdict": evidence.get("upstream_honest_verdict"),
        "rows": deepcopy(evidence.get("rows", empty["rows"])),
        "per_game_results": deepcopy(evidence.get("per_game_results", [])),
        "raw_hash_rows": deepcopy(evidence.get("raw_hash_rows", [])),
        "process_isolation_rows": deepcopy(evidence.get("process_isolation_rows", [])),
        "adapter_access_rows": deepcopy(evidence.get("adapter_access_rows", [])),
        "forbidden_read_rows": deepcopy(evidence.get("forbidden_read_rows", [])),
        "entrypoint_rows": deepcopy(evidence.get("entrypoint_rows", [])),
        "action_provenance_rows": deepcopy(evidence.get("action_provenance_rows", [])),
        "transition_recompute_rows": deepcopy(evidence.get("transition_recompute_rows", [])),
        "removal_replay_rows": deepcopy(evidence.get("removal_replay_rows", [])),
        "causal_credit_rows": deepcopy(evidence.get("causal_credit_rows", [])),
        "level_rows": deepcopy(evidence.get("level_rows", [])),
        "registry_hash_before": evidence.get("registry_hash_before"),
        "registry_hash_after": evidence.get("registry_hash_after"),
        "registry_mutated": bool(evidence.get("registry_mutated", False)),
        "solve_provenance": "development_proxy",
        "solve_claim_made": False,
        "offline_reproduced": False,
        "arc_causal_audit_complete_score": int(evidence.get("arc_causal_audit_complete_score", 0)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(preconditions_checked),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def blocked_artifact(run_date: str, check: str, expected: Any, observed: Any) -> JsonDict:
    """Return a schema-complete terminal block for an unavailable upstream."""

    preconditions = [gate_check(check, expected, observed, passed=False)]
    return build_artifact(
        run_date=run_date,
        duration_s=0.0,
        preconditions_checked=preconditions,
        source_artifact_hashes={},
        evidence={},
        verdict_class="blocked",
        honest_verdict=f"blocked_arc_causal_provenance_audit_{check}",
        inference_substrate_class="blocked_no_run",
    )


def _safe_child(path: Path, root: Path) -> tuple[Path, bool]:
    resolved = path.resolve(strict=False)
    allowed = root.resolve(strict=False)
    return resolved, allowed in resolved.parents


def _parse_jsonl(path: Path) -> tuple[list[JsonDict], str | None]:
    rows: list[JsonDict] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                return [], f"line_{number}_not_object"
            rows.append(dict(value))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return rows, None


def read_bound_traces(
    manifest: Sequence[Mapping[str, Any]], raw_root: Path
) -> tuple[list[JsonDict], dict[str, list[JsonDict]]]:
    """Hash every bound trace before returning any parsed event."""

    canonical = [str(Path(str(row.get("path", ""))).resolve(strict=False)) for row in manifest]
    duplicates = Counter(canonical)
    receipts: list[JsonDict] = []
    events: dict[str, list[JsonDict]] = {}
    for row, canonical_path in zip(manifest, canonical, strict=True):
        arm = str(row.get("arm", ""))
        resolved, path_safe = _safe_child(Path(str(row.get("path", ""))), raw_root)
        exists = resolved.is_file() if path_safe else False
        observed_hash = sha256_path(resolved) if exists else None
        observed_bytes = resolved.stat().st_size if exists else None
        parsed, parse_error = _parse_jsonl(resolved) if exists else ([], "file_unavailable")
        observed_events = len(parsed) if parse_error is None else None
        mismatches: list[str] = []
        if not path_safe:
            mismatches.append("path_outside_raw_root")
        if duplicates[canonical_path] != 1:
            mismatches.append("duplicate_trace_binding")
        if observed_hash != row.get("sha256"):
            mismatches.append("sha256")
        if observed_bytes != row.get("byte_count"):
            mismatches.append("byte_count")
        if observed_events != row.get("event_count"):
            mismatches.append("event_count")
        if parse_error is not None:
            mismatches.append("jsonl_parse")
        passed = not mismatches
        receipts.append(
            {
                "arm": arm,
                "path": str(row.get("path", "")),
                "resolved_path": str(resolved),
                "path_safe": path_safe,
                "expected_sha256": row.get("sha256"),
                "observed_sha256": observed_hash,
                "expected_byte_count": row.get("byte_count"),
                "observed_byte_count": observed_bytes,
                "expected_event_count": row.get("event_count"),
                "observed_event_count": observed_events,
                "parse_error": parse_error,
                "mismatches": mismatches,
                "passed": passed,
            }
        )
        if passed:
            events[arm] = parsed
    return receipts, events


def _events(rows: Sequence[Mapping[str, Any]], kind: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("kind") == kind]


def _source_slice(source: str, start: str, end: str) -> str:
    begin = source.find(start)
    if begin < 0:
        return ""
    finish = source.find(end, begin + len(start))
    return source[begin:] if finish < 0 else source[begin:finish]


def inspect_processes(
    parent_events: Sequence[Mapping[str, Any]], upstream: Mapping[str, Any]
) -> list[JsonDict]:
    """Join each arm identity across raw and projected process receipts."""

    starts = _events(parent_events, "process_started")
    finishes = _events(parent_events, "process_finished")
    phases = _events(parent_events, "phase")
    upstream_rows = list(upstream.get("process_rows") or [])
    raw_pids = [row.get("pid") for row in starts if row.get("role") in ARM_NAMES]
    distinct = len(raw_pids) == len(ARM_NAMES) and len(set(raw_pids)) == len(ARM_NAMES)
    out: list[JsonDict] = []
    for arm in ARM_NAMES:
        arm_starts = [row for row in starts if row.get("role") == arm]
        arm_finishes = [row for row in finishes if row.get("role") == arm]
        arm_phases = [row for row in phases if row.get("phase") == arm]
        projected = [row for row in upstream_rows if row.get("role") == arm]
        start = arm_starts[0] if len(arm_starts) == 1 else {}
        finish = arm_finishes[0] if len(arm_finishes) == 1 else {}
        phase = arm_phases[0] if len(arm_phases) == 1 else {}
        claimed = projected[0] if len(projected) == 1 else {}
        identity = (start.get("pid"), start.get("start_ticks"))
        matches = bool(start) and all(
            (row.get("pid", row.get("process_pid")), row.get("start_ticks", row.get("process_start_ticks")))
            == identity
            for row in (finish, phase, claimed)
        )
        fresh_distinct = bool(
            distinct
            and isinstance(identity[0], int)
            and identity[0] > 0
            and isinstance(identity[1], int)
            and identity[1] >= 0
            and finish.get("fresh_process") is True
        )
        mismatches = []
        if not matches:
            mismatches.append("identity_receipts")
        if not fresh_distinct:
            mismatches.append("fresh_distinct_pid")
        out.append(
            {
                "arm": arm,
                "pid": identity[0],
                "start_ticks": identity[1],
                "start_receipt_count": len(arm_starts),
                "finish_receipt_count": len(arm_finishes),
                "phase_receipt_count": len(arm_phases),
                "upstream_receipt_count": len(projected),
                "identity_receipts_match": matches,
                "fresh_distinct_pid": fresh_distinct,
                "mismatches": mismatches,
                "passed": not mismatches,
            }
        )
    return out


def inspect_adapter_access(
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    producer_source: str,
    adapter_source: str,
    game: str,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Detect adapter code retained in the policy worker after key removal."""

    worker = _source_slice(producer_source, "def _arm_worker", "def _worker_main")
    import_at = worker.find("arc_game_adapters")
    removal_at = worker.find("_BUILDERS.pop")
    source_imports_adapter = import_at >= 0
    import_before_removal = source_imports_adapter and removal_at >= 0 and import_at < removal_at
    target_markers = (f"def _{game}", f"{game.upper()}_", f'"{game}": _{game}')
    target_retained = any(marker in adapter_source for marker in target_markers)
    adapter_rows: list[JsonDict] = []
    read_rows: list[JsonDict] = []
    for arm in ARM_NAMES:
        events = list(traces.get(arm) or [])
        adapter_events = _events(events, "adapter")
        receipt = adapter_events[0] if len(adapter_events) == 1 else {}
        before = list(receipt.get("adapter_keys_before") or [])
        after = list(receipt.get("adapter_keys_after") or [])
        removed = list(receipt.get("removed_adapters") or [])
        expected_after = [key for key in before if key != game]
        registry_removed = bool(
            receipt
            and (
                after == expected_after and removed == [game]
                if arm == "adapter_withheld"
                else after == before and removed == []
            )
        )
        access_events = _events(events, "runtime_access")
        access = access_events[0] if len(access_events) == 1 else {}
        runtime_imports = [str(item) for item in access.get("imports") or []]
        raw_adapter_import = any("arc_game_adapters" in item for item in runtime_imports)
        hidden = arm == "adapter_withheld" and (
            raw_adapter_import or (source_imports_adapter and target_retained)
        )
        mismatches = []
        if not registry_removed:
            mismatches.append("adapter_registry_treatment")
        if hidden:
            mismatches.append("target_adapter_code_retained")
        adapter_rows.append(
            {
                "arm": arm,
                "game": game,
                "adapter_receipt_count": len(adapter_events),
                "registry_key_removed": registry_removed,
                "producer_imports_adapter_module": source_imports_adapter,
                "import_before_registry_removal": import_before_removal,
                "runtime_adapter_import_recorded": raw_adapter_import,
                "target_recipe_symbols_retained": target_retained,
                "mismatches": mismatches,
                "passed": not mismatches,
            }
        )

        reads = [str(item) for item in access.get("filesystem_reads") or []]
        game_marker = f"environment_files/{game}/"
        forbidden = [
            item
            for item in reads
            if game_marker in item.replace("\\", "/").lower()
            or any(marker in item.lower() for marker in _FORBIDDEN_READ_MARKERS)
        ]
        status = "checked" if access else "unavailable"
        read_rows.append(
            {
                "arm": arm,
                "status": status,
                "runtime_access_receipt_count": len(access_events),
                "imports": runtime_imports if access else None,
                "arguments": deepcopy(access.get("arguments")) if access else None,
                "environment": deepcopy(access.get("environment")) if access else None,
                "filesystem_reads": reads if access else None,
                "forbidden_reads": forbidden if access else None,
                "passed": not forbidden if access else None,
            }
        )
    return adapter_rows, read_rows


def inspect_entrypoints(
    parent_events: Sequence[Mapping[str, Any]], producer_source: str
) -> list[JsonDict]:
    """Require raw route receipts and executable factory calls in source."""

    worker = _source_slice(producer_source, "def _arm_worker", "def _worker_main")
    source_factory = "make_carnot_agent(" in worker
    source_policy = "E3AgentPolicy" in worker
    finishes = _events(parent_events, "process_finished")
    out = []
    for arm in ARM_NAMES:
        rows = [row for row in finishes if row.get("role") == arm]
        receipt = rows[0] if len(rows) == 1 else {}
        passed = bool(
            len(rows) == 1
            and receipt.get("entrypoint") == REAL_ENTRYPOINT
            and source_factory
            and source_policy
        )
        out.append(
            {
                "arm": arm,
                "receipt_count": len(rows),
                "observed_entrypoint": receipt.get("entrypoint"),
                "expected_entrypoint": REAL_ENTRYPOINT,
                "source_calls_make_carnot_agent": source_factory,
                "source_references_e3_agent_policy": source_policy,
                "passed": passed,
            }
        )
    return out


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(_HASH_RE.fullmatch(value))


def inspect_actions(traces: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    """Join action receipts without crediting standalone proposals."""

    out: list[JsonDict] = []
    for arm in ARM_NAMES:
        rows = list(traces.get(arm) or [])
        proposals = _events(rows, "proposal")
        verifiers = _events(rows, "verifier")
        actions = _events(rows, "action")
        transitions = _events(rows, "transition")
        rewards = _events(rows, "reward")
        levels = _events(rows, "level")
        proposal_ids = [row.get("proposal_id") for row in proposals]
        action_proposal_ids = [row.get("proposal_id") for row in actions]
        all_ids = list(dict.fromkeys([*proposal_ids, *action_proposal_ids]))
        for identifier in all_ids:
            proposal_matches = [row for row in proposals if row.get("proposal_id") == identifier]
            action_matches = [row for row in actions if row.get("proposal_id") == identifier]
            proposal = proposal_matches[0] if len(proposal_matches) == 1 else {}
            action = action_matches[0] if len(action_matches) == 1 else {}
            action_id = proposal.get("action_id", action.get("action_id"))
            verifier_matches = [row for row in verifiers if row.get("proposal_id") == identifier]
            transition_matches = [row for row in transitions if row.get("action_id") == action_id]
            reward_matches = [row for row in rewards if row.get("action_id") == action_id]
            level_matches = [row for row in levels if row.get("action_id") == action_id]
            verifier = verifier_matches[0] if len(verifier_matches) == 1 else {}
            transition = transition_matches[0] if len(transition_matches) == 1 else {}
            reward = reward_matches[0] if len(reward_matches) == 1 else {}
            level = level_matches[0] if len(level_matches) == 1 else {}
            exact_values = bool(
                proposal
                and action
                and proposal.get("action_id") == action.get("action_id")
                and proposal.get("action") == action.get("action")
                and proposal.get("data") == action.get("data")
            )
            environment_response = bool(
                transition
                and _valid_hash(transition.get("before_hash"))
                and _valid_hash(transition.get("after_hash"))
            )
            verifier_exact = bool(
                len(verifier_matches) == 1
                and verifier.get("verifier") == EXACT_VERIFIER
                and verifier.get("accepted") is True
                and verifier.get("oracle") is False
            )
            transition_exact = bool(
                len(transition_matches) == 1 and transition.get("executed") is True
            )
            reward_exact = bool(
                len(reward_matches) == 1 and reward.get("reward") == transition.get("reward")
            )
            level_exact = bool(
                len(level_matches) == 1
                and level.get("level_before") == transition.get("level_before")
                and level.get("level_after") == transition.get("level_after")
                and level.get("solve_provenance") == "development_proxy"
                and level.get("solve_claim_made") is False
                and level.get("offline_reproduced") is False
            )
            credited = bool(
                len(proposal_matches) == 1
                and len(action_matches) == 1
                and action.get("executed") is True
                and exact_values
                and verifier_exact
                and transition_exact
                and environment_response
                and reward_exact
                and level_exact
            )
            checks = {
                "proposal_count": len(proposal_matches),
                "executed_action_count": sum(row.get("executed") is True for row in action_matches),
                "verifier_count": len(verifier_matches),
                "transition_count": len(transition_matches),
                "reward_count": len(reward_matches),
                "level_count": len(level_matches),
                "exact_action_values": exact_values,
                "exact_verifier": verifier_exact,
                "environment_response_receipt": environment_response,
                "exact_reward": reward_exact,
                "exact_level": level_exact,
            }
            mismatches = [key for key, value in checks.items() if value not in (1, True)]
            out.append(
                {
                    "arm": arm,
                    "proposal_id": identifier,
                    "action_id": action_id,
                    "action": proposal.get("action", action.get("action")),
                    "data": deepcopy(proposal.get("data", action.get("data"))),
                    **checks,
                    "level_before": transition.get("level_before"),
                    "level_after": transition.get("level_after"),
                    "mismatches": mismatches,
                    "credited": credited,
                }
            )
    return out


def _arm_level(action_rows: Sequence[Mapping[str, Any]], arm: str) -> int | None:
    credited = [row for row in action_rows if row.get("arm") == arm and row.get("credited") is True]
    if not credited:
        return None
    starts = [int(row.get("level_before", 0) or 0) for row in credited]
    ends = [int(row.get("level_after", 0) or 0) for row in credited]
    return max(0, max(ends) - min(starts))


def recompute_rows(
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    upstream: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Rebuild arm metrics and parent phase timing with per-unit mismatches."""

    summaries: list[JsonDict] = []
    comparisons: list[JsonDict] = []
    claimed_arms = {str(row.get("arm")): row for row in upstream.get("arm_rows") or []}
    for arm in ARM_NAMES:
        raw = list(traces.get(arm) or [])
        level = _arm_level(actions, arm)
        metrics = {
            "request_count": len(_events(raw, "request_completed")),
            "proposal_count": len(_events(raw, "proposal")),
            "action_count": sum(
                row.get("arm") == arm and row.get("credited") is True for row in actions
            ),
            "transition_count": sum(
                row.get("arm") == arm and row.get("credited") is True for row in actions
            ),
            "levels": level,
        }
        claimed = claimed_arms.get(arm, {})
        mismatches = [key for key, value in metrics.items() if claimed.get(key) != value]
        row = {"unit_type": "arm", "arm": arm, **metrics, "mismatches": mismatches}
        comparisons.append({**row, "upstream": deepcopy(dict(claimed)), "passed": not mismatches})
        summaries.append(row)

    parent = list(traces.get("parent") or [])
    claimed_phases = {str(row.get("phase")): row for row in upstream.get("phase_receipt_rows") or []}
    for raw in _events(parent, "phase"):
        phase = str(raw.get("phase"))
        claimed = claimed_phases.get(phase, {})
        try:
            duration = round(
                (int(raw["ended_monotonic_ns"]) - int(raw["started_monotonic_ns"]))
                / 1_000_000_000,
                6,
            )
        except (KeyError, TypeError, ValueError):
            duration = None
        mismatches = []
        if duration != raw.get("duration_s"):
            mismatches.append("raw_duration_s")
        if claimed and duration != claimed.get("duration_s"):
            mismatches.append("upstream_duration_s")
        if claimed and raw.get("cap_s") != claimed.get("cap_s"):
            mismatches.append("cap_s")
        comparisons.append(
            {
                "unit_type": "phase",
                "phase": phase,
                "recomputed_duration_s": duration,
                "raw_duration_s": raw.get("duration_s"),
                "upstream_duration_s": claimed.get("duration_s"),
                "cap_s": raw.get("cap_s"),
                "mismatches": mismatches,
                "passed": not mismatches,
            }
        )
    return summaries, comparisons


def replay_removals(
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    actions: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Replay a removable score only when all deterministic inputs exist."""

    replay_rows: list[JsonDict] = []
    credit_rows: list[JsonDict] = []
    for action in actions:
        arm = str(action.get("arm"))
        action_id = action.get("action_id")
        proposal_id = action.get("proposal_id")
        if action.get("credited") is not True:
            replay = {
                "arm": arm,
                "proposal_id": proposal_id,
                "action_id": action_id,
                "signal": None,
                "status": "not_executed",
                "selected_action_original": None,
                "selected_action_without_signal": None,
                "action_changed": None,
                "reason": "proposal_has_no_complete_executed_chain",
            }
        else:
            candidates = [
                row
                for row in _events(list(traces.get(arm) or []), "selection_replay_input")
                if row.get("action_id") == action_id
            ]
            receipt = candidates[0] if len(candidates) == 1 else {}
            rows = receipt.get("candidates") if isinstance(receipt.get("candidates"), list) else []
            complete = bool(
                len(candidates) == 1
                and receipt.get("selection_rule") == "max_score_then_order"
                and receipt.get("signal") in {"forecast", "verifier_routing"}
                and rows
                and all(
                    isinstance(row, Mapping)
                    and isinstance(row.get("score"), (int, float))
                    and not isinstance(row.get("score"), bool)
                    and isinstance(row.get("signal_contribution"), (int, float))
                    and not isinstance(row.get("signal_contribution"), bool)
                    for row in rows
                )
            )
            original = {"action": action.get("action"), "data": deepcopy(action.get("data"))}
            if complete and receipt.get("selected_action") == original:
                best_index = max(
                    range(len(rows)),
                    key=lambda index: (
                        float(rows[index]["score"]) - float(rows[index]["signal_contribution"]),
                        -index,
                    ),
                )
                selected = {
                    "action": rows[best_index].get("action"),
                    "data": deepcopy(rows[best_index].get("data")),
                }
                replay = {
                    "arm": arm,
                    "proposal_id": proposal_id,
                    "action_id": action_id,
                    "signal": receipt.get("signal"),
                    "status": "replayed",
                    "selected_action_original": original,
                    "selected_action_without_signal": selected,
                    "action_changed": selected != original,
                    "reason": None,
                }
            else:
                replay = {
                    "arm": arm,
                    "proposal_id": proposal_id,
                    "action_id": action_id,
                    "signal": receipt.get("signal"),
                    "status": "unavailable",
                    "selected_action_original": original,
                    "selected_action_without_signal": None,
                    "action_changed": None,
                    "reason": "missing_original_input_signal_candidate_set_or_selection_rule",
                }
        replay_rows.append(replay)
        causal = bool(replay["status"] == "replayed" and replay["action_changed"] is True)
        level_before = action.get("level_before")
        level_after = action.get("level_after")
        credited_delta = (
            max(0, int(level_after or 0) - int(level_before or 0))
            if action.get("credited") is True
            else 0
        )
        credit_rows.append(
            {
                "arm": arm,
                "proposal_id": proposal_id,
                "action_id": action_id,
                "executed_provenance_credit": action.get("credited") is True,
                "removal_replay_status": replay["status"],
                "causal_credit": causal,
                "causal_effect": replay["action_changed"] if replay["status"] == "replayed" else None,
                "credited_level_delta": credited_delta,
            }
        )
    return replay_rows, credit_rows


def _source_hashes(repo_root: Path, upstream_path: Path, raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    paths = {
        "upstream_artifact": upstream_path,
        PRODUCER_RELATIVE_PATH.as_posix(): repo_root / PRODUCER_RELATIVE_PATH,
        ADAPTER_RELATIVE_PATH.as_posix(): repo_root / ADAPTER_RELATIVE_PATH,
        REGISTRY_RELATIVE_PATH.as_posix(): repo_root / REGISTRY_RELATIVE_PATH,
    }
    out = {name: sha256_path(path) for name, path in paths.items()}
    for index, row in enumerate(raw_rows):
        out[f"raw_trace_{index}:{row.get('arm')}"] = row.get("observed_sha256")
    return out


def _source_integrity(
    repo_root: Path, upstream: Mapping[str, Any]
) -> tuple[str, str, list[str]]:
    producer_path = repo_root / PRODUCER_RELATIVE_PATH
    adapter_path = repo_root / ADAPTER_RELATIVE_PATH
    producer = producer_path.read_text(encoding="utf-8") if producer_path.is_file() else ""
    adapter = adapter_path.read_text(encoding="utf-8") if adapter_path.is_file() else ""
    expected = dict(upstream.get("source_artifact_hashes") or {})
    mismatches = []
    for relative, path in ((PRODUCER_RELATIVE_PATH, producer_path), (ADAPTER_RELATIVE_PATH, adapter_path)):
        claimed = expected.get(relative.as_posix())
        if claimed is not None and claimed != sha256_path(path):
            mismatches.append(relative.as_posix())
    return producer, adapter, mismatches


def _level_rows(traces: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    out = []
    for arm in ARM_NAMES:
        for row in _events(list(traces.get(arm) or []), "level"):
            passed = bool(
                row.get("solve_provenance") == "development_proxy"
                and row.get("solve_claim_made") is False
                and row.get("offline_reproduced") is False
            )
            out.append({**deepcopy(dict(row)), "provenance_passed": passed})
    return out


def _upstream_projection_mismatches(
    upstream: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    if not rows or any(row.get("levels") is None for row in rows):
        return []
    by_arm = {str(row["arm"]): row for row in rows}
    withheld = int(by_arm[ARM_NAMES[0]]["levels"])
    control = int(by_arm[ARM_NAMES[1]]["levels"])
    expected_class = "positive" if withheld > 0 else "null"
    expected = {
        "withheld_levels": withheld,
        "control_levels": control,
        "level_delta": withheld - control,
        "verdict_class": expected_class,
    }
    return [key for key, value in expected.items() if upstream.get(key) != value]


def audit_upstream(
    repo_root: Path,
    upstream_path: Path,
    *,
    run_date: str,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> JsonDict:
    """Audit one Exp7127 artifact and all raw traces that it binds."""

    started = time.monotonic()
    if not upstream_path.is_file():
        return blocked_artifact(run_date, "upstream_artifact_present", True, False)
    try:
        upstream_value = json.loads(upstream_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return blocked_artifact(run_date, "upstream_artifact_readable", True, type(exc).__name__)
    if not isinstance(upstream_value, Mapping):
        return blocked_artifact(run_date, "upstream_artifact_object", True, False)
    upstream = dict(upstream_value)
    upstream_class = upstream.get("verdict_class")
    upstream_honest = upstream.get("honest_verdict")
    manifest = list(upstream.get("raw_trace_manifest") or [])
    producer, adapter, source_mismatches = _source_integrity(repo_root, upstream)
    registry_path = repo_root / REGISTRY_RELATIVE_PATH
    registry_current = sha256_path(registry_path)
    source_claims = dict(upstream.get("source_artifact_hashes") or {})
    registry_before = source_claims.get(REGISTRY_RELATIVE_PATH.as_posix())
    producer_after = source_claims.get("registry_hash_after")
    registry_mutated = bool(
        registry_before
        and (producer_after != registry_before or registry_current != registry_before)
    )

    if upstream_class == "blocked" and not manifest:
        blockage = upstream.get("gate_check_summary")
        blockage_complete = bool(
            isinstance(blockage, Mapping)
            and blockage.get("failed_check")
            and "expected_value" in blockage
            and "observed_value" in blockage
            and isinstance(upstream_honest, str)
            and upstream_honest.startswith("blocked_")
        )
        checks = [
            gate_check("upstream_artifact_present", True, True),
            gate_check("upstream_terminal_blockage_complete", True, blockage_complete),
        ]
        verdict = "null" if blockage_complete else "disqualified"
        honest = (
            "complete_null_upstream_blockage_audited_no_causal_evidence"
            if blockage_complete
            else "complete_disqualified_upstream_blockage_provenance"
        )
        evidence = {
            "upstream_verdict_class": upstream_class,
            "upstream_honest_verdict": upstream_honest,
            "registry_hash_before": registry_before,
            "registry_hash_after": producer_after,
            "registry_mutated": registry_mutated,
            "arc_causal_audit_complete_score": 1,
        }
        return build_artifact(
            run_date=run_date,
            duration_s=time.monotonic() - started,
            preconditions_checked=checks,
            source_artifact_hashes=_source_hashes(repo_root, upstream_path, []),
            evidence=evidence,
            verdict_class=verdict,
            honest_verdict=honest,
        )

    raw_rows, traces = read_bound_traces(manifest, raw_root)
    parent = list(traces.get("parent") or [])
    game = str((upstream.get("selected_game") or {}).get("game") or "")
    process_rows = inspect_processes(parent, upstream)
    adapter_rows, forbidden_rows = inspect_adapter_access(traces, producer, adapter, game)
    entrypoint_rows = inspect_entrypoints(parent, producer)
    action_rows = inspect_actions(traces)
    rows, recomputed_rows = recompute_rows(traces, upstream, action_rows)
    removal_rows, causal_rows = replay_removals(traces, action_rows)
    levels = _level_rows(traces)
    projection_mismatches = _upstream_projection_mismatches(upstream, rows)
    arm_levels = {str(row["arm"]): row.get("levels") for row in rows}
    per_game = [
        {
            "game": game,
            "withheld_levels": arm_levels.get(ARM_NAMES[0]),
            "control_levels": arm_levels.get(ARM_NAMES[1]),
            "level_delta": (
                int(arm_levels[ARM_NAMES[0]]) - int(arm_levels[ARM_NAMES[1]])
                if all(arm_levels.get(arm) is not None for arm in ARM_NAMES)
                else None
            ),
            "upstream_verdict_class": upstream_class,
            "recomputed_upstream_terminal_class": (
                "positive"
                if arm_levels.get(ARM_NAMES[0]) is not None
                and int(arm_levels[ARM_NAMES[0]]) > 0
                else "null"
                if all(arm_levels.get(arm) is not None for arm in ARM_NAMES)
                else "unavailable"
            ),
            "projection_mismatches": projection_mismatches,
            "solve_provenance": "development_proxy",
            "solve_claim_made": False,
            "offline_reproduced": False,
        }
    ]

    raw_complete = bool(manifest) and len(raw_rows) == len(manifest) and all(
        row["passed"] for row in raw_rows
    )
    missing_raw = any(row["observed_sha256"] is None and row["path_safe"] for row in raw_rows)
    unsafe_raw = any(not row["path_safe"] for row in raw_rows)
    process_complete = len(process_rows) == len(ARM_NAMES) and all(row["passed"] for row in process_rows)
    adapter_clean = len(adapter_rows) == len(ARM_NAMES) and all(row["passed"] for row in adapter_rows)
    entrypoints_clean = len(entrypoint_rows) == len(ARM_NAMES) and all(
        row["passed"] for row in entrypoint_rows
    )
    arms_executed = all(
        any(row.get("arm") == arm and row.get("credited") is True for row in action_rows)
        for arm in ARM_NAMES
    )
    recompute_clean = not projection_mismatches and all(row["passed"] for row in recomputed_rows)
    provenance_clean = bool(levels) and all(row["provenance_passed"] for row in levels)
    source_clean = bool(producer and adapter) and not source_mismatches
    registry_available = bool(registry_before and producer_after and registry_current)
    checks = [
        gate_check("upstream_artifact_present", True, True),
        gate_check("raw_trace_integrity", True, raw_complete),
        gate_check("process_isolation", True, process_complete),
        gate_check("adapter_access_clean", True, adapter_clean),
        gate_check("entrypoint_receipts", True, entrypoints_clean),
        gate_check("executed_action_chain_per_arm", True, arms_executed),
        gate_check("independent_recompute_matches", True, recompute_clean),
        gate_check("source_hashes_match", True, source_clean),
        gate_check("registry_hashes_available", True, registry_available),
        gate_check("registry_mutated", False, registry_mutated),
        gate_check("level_provenance", True, provenance_clean),
    ]
    disqualified = bool(
        unsafe_raw
        or (not raw_complete and not missing_raw)
        or not process_complete
        or not adapter_clean
        or not entrypoints_clean
        or not recompute_clean
        or not source_clean
        or registry_mutated
        or (levels and not provenance_clean)
        or upstream_class == "disqualified"
    )
    blocked = bool(
        not disqualified
        and (
            missing_raw
            or not manifest
            or not arms_executed
            or not registry_available
            or not levels
        )
    )
    causal_supported = any(row["causal_credit"] for row in causal_rows)
    if disqualified:
        verdict = "disqualified"
        failed = next((row["check"] for row in checks if row["passed"] is not True), "upstream")
        honest = f"complete_disqualified_arc_causal_provenance_{failed}"
    elif blocked:
        verdict = "blocked"
        failed = next((row["check"] for row in checks if row["passed"] is not True), "evidence")
        honest = f"blocked_arc_causal_provenance_audit_{failed}"
    elif causal_supported:
        verdict = "positive"
        honest = "complete_positive_independent_action_removal_effect"
    else:
        verdict = "null"
        honest = "complete_null_no_supported_action_removal_effect"
    evidence = {
        "upstream_verdict_class": upstream_class,
        "upstream_honest_verdict": upstream_honest,
        "rows": rows,
        "per_game_results": per_game,
        "raw_hash_rows": raw_rows,
        "process_isolation_rows": process_rows,
        "adapter_access_rows": adapter_rows,
        "forbidden_read_rows": forbidden_rows,
        "entrypoint_rows": entrypoint_rows,
        "action_provenance_rows": action_rows,
        "transition_recompute_rows": recomputed_rows,
        "removal_replay_rows": removal_rows,
        "causal_credit_rows": causal_rows,
        "level_rows": levels,
        "registry_hash_before": registry_before,
        "registry_hash_after": producer_after,
        "registry_mutated": registry_mutated,
        "arc_causal_audit_complete_score": 0 if blocked else 1,
    }
    return build_artifact(
        run_date=run_date,
        duration_s=time.monotonic() - started,
        preconditions_checked=checks,
        source_artifact_hashes=_source_hashes(repo_root, upstream_path, raw_rows),
        evidence=evidence,
        verdict_class=verdict,
        honest_verdict=honest,
        inference_substrate_class="blocked_no_run" if blocked else "aggregation",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, terminal class, non-claims, and self-hash."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_artifact_fields_mismatch")
    if set(artifact.get("field_principles") or {}) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    verdict = artifact.get("verdict_class")
    if verdict not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    honest = artifact.get("honest_verdict")
    if verdict in VERDICT_PREFIXES and not (
        isinstance(honest, str) and honest.startswith(VERDICT_PREFIXES[str(verdict)])
    ):
        errors.append("verdict_prefix_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("inference_substrate_class") not in {"aggregation", "blocked_no_run"}:
        errors.append("inference_substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance_mismatch")
    if artifact.get("solve_claim_made") is not False:
        errors.append("solve_claim_made_must_be_false")
    if artifact.get("offline_reproduced") is not False:
        errors.append("offline_reproduced_must_be_false")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    score = artifact.get("arc_causal_audit_complete_score")
    if not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1}:
        errors.append("arc_causal_audit_complete_score_invalid")
    if artifact.get("registry_mutated") is True and verdict != "disqualified":
        errors.append("registry_mutation_requires_disqualified")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def run_audit(
    *,
    repo_root: Path,
    upstream_path: Path,
    output_path: Path,
    run_date: str,
    raw_root: Path = DEFAULT_RAW_ROOT,
) -> JsonDict:
    """Run the audit once and atomically publish its terminal artifact."""

    artifact = audit_upstream(
        repo_root.resolve(), upstream_path.resolve(), run_date=run_date, raw_root=raw_root.resolve()
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - a self-invalid artifact is a programming error.
        raise RuntimeError(f"Exp7128 artifact validation failed: {errors}")
    atomic_write_json(output_path.resolve(), artifact, allow_override=False)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI exercised by task command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=time.strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--upstream", type=Path, default=ROOT / UPSTREAM_RELATIVE_PATH)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return int(bool(errors))
    artifact = run_audit(
        repo_root=ROOT,
        upstream_path=args.upstream,
        output_path=args.output,
        run_date=str(args.date),
        raw_root=args.raw_root,
    )
    print(
        json.dumps(
            {"artifact": str(args.output.resolve()), "verdict": artifact["honest_verdict"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
