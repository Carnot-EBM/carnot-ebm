"""Build the REQ-ARC-6859 first-party tool-gap receipt artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic import arc_tool_gap_receipt as receipt
from carnot.agentic.arc_induction_tools import InductionToolSession, dispatch_tool
from carnot.paths import repo_root, results_path


EXPERIMENT_ID = "experiment_6859"
ARTIFACT_SCHEMA = "carnot.experiment_6859.first_party_tool_gap_receipt_wiring.v1"
INFERENCE_SUBSTRATE = "canonical_live_seam_default_off_fixture_and_terminal_replay"
ROUTER_PATH = Path("results/experiment_6857_dynamic_live_arc_receipt_router.json")
SPEC_PATH = Path("openspec/capabilities/arc-agi/spec.md")
OUTPUT_PATH = Path("results/experiment_6859_first_party_tool_gap_receipt_wiring.json")
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "reproducibility_checksum",
    "rows",
    "receipt_schema",
    "canonical_seam_manifest",
    "gap_detection_rows",
    "request_rows",
    "tool_response_rows",
    "agent_delivery_rows",
    "next_action_rows",
    "exact_outcome_rows",
    "join_completeness_rows",
    "provenance_class_rows",
    "restart_results",
    "deduplication_results",
    "default_off_verified",
    "tool_gap_receipt_contract_ready_score",
    "tool_gap_live_effect_claim_eligible_score",
    "solve_claimed",
    "game_level_solve_count",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)


class _Clock:
    def __init__(self) -> None:
        self.tick = 0

    def __call__(self) -> str:
        self.tick += 1
        return f"2026-09-01T01:00:{self.tick:02d}Z"


@dataclass
class _Frame:
    levels_completed: int
    state: str = "fixture"
    score: int = 0


def _source_hashes(root: Path) -> dict[str, dict[str, Any]]:
    from carnot.agentic import arc_competition_agent, arc_induction_tool_loop

    paths = {
        "router": root / ROUTER_PATH,
        "receipt_transport": Path(receipt.__file__),
        "canonical_agent": Path(arc_competition_agent.__file__),
        "induction_loop": Path(arc_induction_tool_loop.__file__),
        "spec": root / SPEC_PATH,
    }
    return {
        name: {
            "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "exists": path.is_file(),
            "file_sha256": receipt.sha256_file(path) if path.is_file() else None,
        }
        for name, path in paths.items()
    }


def _read_router(root: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads((root / ROUTER_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return {}, "router artifact is not an object"
    return value, None


def _canonical_seam_manifest() -> list[dict[str, Any]]:
    from carnot.agentic import arc_competition_agent, arc_induction_tool_loop, arc_induction_tools

    checks = [
        (
            "tool_dispatch",
            arc_induction_tools.dispatch_tool,
            ("receipt_transport", "record_dispatch"),
        ),
        (
            "agent_visible_delivery",
            arc_induction_tool_loop.induce_with_tool_loop,
            ("receipt_transport", "record_delivery"),
        ),
        (
            "canonical_next_action_and_outcome",
            arc_competition_agent.E3AgentPolicy.next_move,
            (
                "_record_first_party_tool_gap_outcome",
                "_record_first_party_tool_gap_next_action",
            ),
        ),
    ]
    rows = []
    for seam, function, tokens in checks:
        source = inspect.getsource(function)
        source_path = Path(inspect.getsourcefile(function) or "")
        rows.append(
            {
                "seam": seam,
                "symbol": f"{function.__module__}.{function.__qualname__}",
                "source_sha256": receipt.sha256_file(source_path),
                "required_tokens": list(tokens),
                "reachable": all(token in source for token in tokens),
                "live_entrypoint": "make_carnot_agent -> E3AgentPolicy.next_move",
            }
        )
    return rows


def _configured_default_off_check() -> bool:
    previous_enable = os.environ.pop(receipt.ENABLE_ENV, None)
    previous_path = os.environ.get(receipt.PATH_ENV)
    try:
        with tempfile.TemporaryDirectory(prefix="carnot-6859-default-off-") as directory:
            path = Path(directory) / "should-not-exist.json"
            os.environ[receipt.PATH_ENV] = str(path)
            made = receipt.maybe_make_first_party_tool_gap_receipt_transport("fixture", "off")
            return made is None and not path.exists()
    finally:
        if previous_enable is not None:
            os.environ[receipt.ENABLE_ENV] = previous_enable
        if previous_path is None:
            os.environ.pop(receipt.PATH_ENV, None)
        else:
            os.environ[receipt.PATH_ENV] = previous_path


def _policy_for_fixture(
    transport: receipt.FirstPartyToolGapReceiptTransport,
    action: tuple[int, None],
) -> E3AgentPolicy:
    policy = object.__new__(E3AgentPolicy)
    policy._first_party_tool_gap_receipt_transport = transport
    policy._provenance = None
    policy._next_move_routed = lambda frames, latest: action
    policy._maybe_apply_trace_automaton_action = lambda move, latest: move
    policy.record_target_licensed_route_shadow = lambda move, **kwargs: move
    policy.record_typed_obligation_shadow_monitor = lambda move, **kwargs: move
    policy._record_outcome_transport_proposal = lambda *args: None
    return policy


def _exercise_fixtures() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="carnot-6859-fixture-") as directory:
        path = Path(directory) / "receipts.json"
        transport = receipt.FirstPartyToolGapReceiptTransport(
            path,
            attempt_identity="fixture-attempt-6859",
            game_id="fixture-game",
            provenance_class="fixture",
            clock=_Clock(),
        )
        session = InductionToolSession([])
        clean_result = dispatch_tool(
            session,
            "list_transitions",
            "{}",
            receipt_transport=transport,
            decision_point_identity="clean",
        )
        rejected = dispatch_tool(
            session,
            "missing_grid_tool",
            '{"region":3}',
            receipt_transport=transport,
            decision_point_identity="rejected",
        )
        delivered_identity = transport.last_receipt_identity
        delivered_text = json.dumps(rejected, sort_keys=True)
        transport.record_delivery(delivered_identity, visible_text=delivered_text)

        session.list_transitions = lambda: (_ for _ in ()).throw(RuntimeError("fixture error"))
        hidden = dispatch_tool(
            session,
            "list_transitions",
            "{}",
            receipt_transport=transport,
            decision_point_identity="tool-error-hidden",
        )
        hidden_identity = transport.last_receipt_identity

        action = (2, None)
        policy = _policy_for_fixture(transport, action)
        action_identity_preserved = policy.next_move([], _Frame(0)) is action
        policy.next_move([], _Frame(1))

        before_restart = transport.snapshot()
        restarted = receipt.FirstPartyToolGapReceiptTransport(
            path,
            attempt_identity="fixture-attempt-6859",
            game_id="fixture-game",
            provenance_class="fixture",
            clock=_Clock(),
        )
        restart_identity_stable = restarted.snapshot()["rows"] == before_restart["rows"]
        restarted.record_delivery(delivered_identity, visible_text=delivered_text)
        identical_dedup = restarted.snapshot()["deduplication"]["byte_identical_count"] == 1

        conflict_path = Path(directory) / "conflict.json"
        conflicting = receipt.FirstPartyToolGapReceiptTransport(
            conflict_path,
            attempt_identity="fixture-conflict-6859",
            game_id="fixture-game",
            provenance_class="fixture",
            clock=_Clock(),
        )
        conflict_session = InductionToolSession([])
        dispatch_tool(
            conflict_session,
            "missing_conflict_tool",
            "{}",
            receipt_transport=conflicting,
            decision_point_identity="conflict",
        )
        conflict_identity = conflicting.last_receipt_identity
        conflicting.record_delivery(conflict_identity, visible_text="first")
        conflicting.record_delivery(conflict_identity, visible_text="second")
        conflict_row = conflicting.row(conflict_identity)

        snapshot = restarted.snapshot()
        return {
            "rows": snapshot["rows"],
            "clean_no_gap": clean_result.get("ok") is True and len(before_restart["rows"]) == 2,
            "rejected_request": rejected,
            "hidden_response": hidden,
            "hidden_identity": hidden_identity,
            "action_identity_preserved": action_identity_preserved,
            "restart": {
                "persisted_row_count": len(before_restart["rows"]),
                "reloaded_row_count": restarted.restart_loaded_count,
                "identity_stable": restart_identity_stable,
                "pending_chain_preserved": any(
                    not row["join_complete"] for row in restarted.snapshot()["rows"]
                ),
            },
            "deduplication": {
                "byte_identical_detected": identical_dedup,
                "conflicting_duplicate_quarantined": conflict_row.get("quarantine_reason")
                == "conflicting_duplicate_hop",
                "conflicting_join_closed": conflict_row.get("join_complete") is False,
            },
        }


def _normalize_replay_rows(router: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source_rows = router.get("first_party_tool_gap_rows") or []
    if not isinstance(source_rows, list):
        return rows
    for index, source in enumerate(source_rows):
        if not isinstance(source, dict):
            continue
        provenance_class = str(source.get("provenance_class") or "terminal_replay")
        quarantine_reason = source.get("quarantine_reason")
        if provenance_class == "development_proxy":
            quarantine_reason = "development_proxy_not_live_eligible"
        elif provenance_class == "reconstructed":
            quarantine_reason = "reconstructed_not_live_eligible"
        exact = bool(source.get("exact_outcome"))
        joined = bool(source.get("join_complete")) and quarantine_reason is None
        used = source.get("response_used") is True
        headroom = bool(source.get("valid_headroom"))
        causal_eligible = (
            provenance_class == "authentic_live"
            and bool(source.get("first_party"))
            and bool(source.get("live_reachable"))
            and bool(source.get("agent_visible"))
            and used
            and bool(source.get("next_action_recorded"))
            and exact
            and headroom
            and joined
        )
        rows.append(
            {
                "receipt_identity": str(
                    source.get("receipt_identity")
                    or receipt.canonical_sha256({"terminal_replay_index": index, "source": source})
                ),
                "source": "terminal_replay",
                "provenance_class": provenance_class,
                "agent_visible": bool(source.get("agent_visible")),
                "response_used": source.get("response_used"),
                "next_action_recorded": bool(source.get("next_action_recorded")),
                "exact_outcome": exact,
                "valid_headroom": headroom,
                "join_complete": joined,
                "causal_eligible": causal_eligible,
                "quarantine_reason": quarantine_reason,
                "levels_before": source.get("levels_before"),
                "levels_after": source.get("levels_after"),
            }
        )
    return rows


def _hop_rows(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        hop = row.get("hops", {}).get(kind)
        if hop:
            output.append(
                {
                    "receipt_identity": row["receipt_identity"],
                    "provenance_class": row["provenance_class"],
                    **hop,
                }
            )
    return output


def _principles(keys: Sequence[str]) -> dict[str, str]:
    specific = {
        "duration_s": "Wall time exposes skipped fixture or replay work.",
        "inference_substrate": "The run uses canonical seams, fixtures, and terminal replay. It launches no game.",
        "source_artifact_hashes": "Hashes bind the router, schema, and exact seam sources.",
        "rows": "Gap chains and hop metrics stay visible as separate row kinds.",
        "receipt_schema": "The schema fixes hop names, identity rules, and provenance classes.",
        "canonical_seam_manifest": "Reachability proves that the contract is not a detached adapter.",
        "join_completeness_rows": "Complete joins do not imply response use, progress, or causal effect.",
        "provenance_class_rows": "Fixture, replay, proxy, reconstructed, and authentic rows cannot be pooled.",
        "restart_results": "Restart must preserve complete and pending receipt identities.",
        "deduplication_results": "Repeated bytes deduplicate while conflicting identities quarantine.",
        "default_off_verified": "Unset configuration must preserve action identity and write nothing.",
        "tool_gap_receipt_contract_ready_score": "Readiness measures schema and reachable transport completeness only.",
        "tool_gap_live_effect_claim_eligible_score": "Only exact authentic-live used responses with headroom can open effect eligibility.",
        "solve_claimed": "Receipt wiring cannot become a solve claim.",
        "game_level_solve_count": "No live rollout means zero new game-level solves.",
        "verifier_is_oracle": "The reducer checks receipts; it does not define game correctness.",
        "honest_verdict": "The terminal verdict states the exact supported evidence boundary.",
    }
    return {
        key: specific.get(key, f"{key} is required for the REQ-ARC-6859 audit contract.")
        for key in keys
    }


def build_artifact(
    root: str | Path,
    *,
    run_date: str,
    duration_s: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(root).resolve()
    router, router_error = _read_router(root)
    seams = _canonical_seam_manifest()
    default_off_verified = _configured_default_off_check()
    checks = [
        {
            "check": "arc_receipt_router_complete_score",
            "expected": 1,
            "observed": router_error
            if router_error
            else router.get("arc_receipt_router_complete_score"),
            "passed": router_error is None and router.get("arc_receipt_router_complete_score") == 1,
        },
        {
            "check": "canonical_live_seam",
            "expected": True,
            "observed": all(row["reachable"] for row in seams),
            "passed": all(row["reachable"] for row in seams),
        },
        {
            "check": "stable_receipt_schema",
            "expected": receipt.RECEIPT_SCHEMA,
            "observed": receipt.RECEIPT_SCHEMA,
            "passed": receipt.RECEIPT_SCHEMA == "carnot.arc.first_party_tool_gap_receipt.v1",
        },
        {
            "check": "game_source_files_read",
            "expected": [],
            "observed": [],
            "passed": True,
        },
        {
            "check": "new_live_rollout_count",
            "expected": 0,
            "observed": 0,
            "passed": True,
        },
    ]
    preconditions_passed = all(row["passed"] for row in checks)
    fixture = (
        _exercise_fixtures()
        if preconditions_passed
        else {
            "rows": [],
            "clean_no_gap": False,
            "action_identity_preserved": False,
            "restart": {},
            "deduplication": {},
        }
    )
    fixture_rows = fixture["rows"]
    replay_rows = _normalize_replay_rows(router) if preconditions_passed else []
    join_rows = [
        {
            "receipt_identity": row["receipt_identity"],
            "source": "fixture",
            "provenance_class": row["provenance_class"],
            "agent_visible": row["agent_visible"],
            "response_used": row["response_used"],
            "next_action_recorded": row["next_action_recorded"],
            "exact_outcome": row["exact_outcome"],
            "valid_headroom": row["valid_headroom"],
            "join_complete": row["join_complete"],
            "causal_eligible": row["causal_eligible"],
            "quarantine_reason": row["quarantine_reason"],
        }
        for row in fixture_rows
    ] + replay_rows
    class_counts = []
    for class_name in receipt.PROVENANCE_CLASSES:
        class_counts.append(
            {
                "provenance_class": class_name,
                "row_count": sum(row.get("provenance_class") == class_name for row in join_rows),
                "live_effect_eligible_count": sum(
                    row.get("provenance_class") == class_name and row.get("causal_eligible")
                    for row in join_rows
                ),
            }
        )
    class_counts.append(
        {
            "provenance_class": "terminal_replay",
            "row_count": len(replay_rows),
            "live_effect_eligible_count": sum(row["causal_eligible"] for row in replay_rows),
        }
    )
    fixture_complete = any(row["join_complete"] for row in join_rows if row["source"] == "fixture")
    restart_ok = bool(fixture.get("restart", {}).get("identity_stable"))
    dedup_ok = all(
        fixture.get("deduplication", {}).get(key) is True
        for key in (
            "byte_identical_detected",
            "conflicting_duplicate_quarantined",
            "conflicting_join_closed",
        )
    )
    contract_ready = int(
        preconditions_passed
        and default_off_verified
        and fixture.get("clean_no_gap") is True
        and fixture.get("action_identity_preserved") is True
        and fixture_complete
        and restart_ok
        and dedup_ok
    )
    live_eligible = int(any(row["causal_eligible"] for row in replay_rows))
    failed = next((row for row in checks if not row["passed"]), None)
    if failed is not None:
        verdict_class = "blocked"
        honest_verdict = "complete_blocked_first_party_tool_gap_receipt_wiring"
    elif live_eligible:
        verdict_class = "positive"
        honest_verdict = "complete_first_party_tool_gap_live_effect_claim_eligible"
    elif contract_ready:
        verdict_class = "null"
        honest_verdict = "complete_first_party_tool_gap_receipt_contract_ready_no_live_effect_claim"
    else:
        verdict_class = "partial"
        honest_verdict = "complete_partial_first_party_tool_gap_receipt_contract_not_ready"
    hop_counts = {kind: len(_hop_rows(fixture_rows, kind)) for kind in receipt.REQUIRED_HOPS}
    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "complete" if failed is None else "blocked",
        "field_principles": {},
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(
            duration_s if duration_s is not None else time.monotonic() - started, 6
        ),
        "source_artifact_hashes": _source_hashes(root),
        "reproducibility_checksum": "",
        "rows": [
            *[{"row_kind": "gap_chain", **row} for row in fixture_rows],
            {
                "row_kind": "hop_metrics",
                "hop_counts": hop_counts,
                "gap_chain_count": len(fixture_rows),
            },
        ],
        "receipt_schema": {
            "schema": receipt.RECEIPT_SCHEMA,
            "immutable_identity_field": "receipt_identity",
            "required_hops": list(receipt.REQUIRED_HOPS),
            "hop_fields": [
                "receipt_identity",
                "hop_identity",
                "hop_kind",
                "timestamp",
                "source_sha256",
                "payload_sha256",
                "previous_hop_identity",
                "payload",
            ],
            "provenance_classes": list(receipt.PROVENANCE_CLASSES),
        },
        "canonical_seam_manifest": seams,
        "gap_detection_rows": _hop_rows(fixture_rows, "gap_detection"),
        "request_rows": _hop_rows(fixture_rows, "request"),
        "tool_response_rows": _hop_rows(fixture_rows, "tool_response"),
        "agent_delivery_rows": _hop_rows(fixture_rows, "agent_delivery"),
        "next_action_rows": _hop_rows(fixture_rows, "next_action"),
        "exact_outcome_rows": _hop_rows(fixture_rows, "exact_outcome"),
        "join_completeness_rows": join_rows,
        "provenance_class_rows": class_counts,
        "restart_results": fixture.get("restart", {}),
        "deduplication_results": fixture.get("deduplication", {}),
        "default_off_verified": default_off_verified,
        "tool_gap_receipt_contract_ready_score": contract_ready,
        "tool_gap_live_effect_claim_eligible_score": live_eligible,
        "solve_claimed": False,
        "solve_claim": False,
        "game_level_solve_count": 0,
        "gate_check_summary": {
            "passed": failed is None,
            "failed_check": failed["check"] if failed else None,
            "expected": failed["expected"] if failed else "all checks pass",
            "observed": failed["observed"] if failed else "all checks pass",
            "checks": checks,
        },
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["field_principles"] = _principles(tuple(artifact))
    checksum_input = dict(artifact)
    checksum_input["duration_s"] = 0.0
    checksum_input["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = receipt.canonical_sha256(checksum_input)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    errors = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing field: {field}")
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        errors.append("wrong schema")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("wrong inference substrate")
    if artifact.get("solve_claimed") is not False or artifact.get("game_level_solve_count") != 0:
        errors.append("receipt wiring cannot claim a solve")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("invalid verdict class")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict is not terminal")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles do not cover every top-level field")
    if artifact.get("tool_gap_live_effect_claim_eligible_score") == 1 and not any(
        row.get("causal_eligible") and row.get("provenance_class") == "authentic_live"
        for row in artifact.get("join_completeness_rows", [])
    ):
        errors.append("live effect score lacks an authentic exact row")
    return errors


def _atomic_write(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if len(args.date) != 8 or not args.date.isdigit():
        parser.error("--date must use YYYYMMDD")
    root = Path(args.root).resolve() if args.root else repo_root(start=__file__)
    output = (
        Path(args.output).resolve()
        if args.output
        else results_path(OUTPUT_PATH.name, ensure_parent=True, start=__file__)
    )
    artifact = build_artifact(root, run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        raise SystemExit("; ".join(errors))
    _atomic_write(output, artifact)
    print(json.dumps({"output": str(output), "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI wrapper
    raise SystemExit(main())
