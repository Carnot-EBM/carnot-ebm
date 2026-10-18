"""Inspect immutable live receipts before ordering events. REQ-REPORT-7936-IDENTITY.

Raw producer hashes authenticate the event fields. Content identity permits
inspection on the same day, but it cannot supply a missing scientific clock.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone, UTC
import json
from pathlib import Path
import re
from typing import Any

from carnot.agentic.arc_supervisor_refinement import classify_receipt, extract_rows, wilson_bounds
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


def event_order(row: dict[str, Any], cutoff: dict[str, Any], current: str) -> str:
    """Only clocks inside authenticated raw bytes can establish chronology."""
    timestamp = row.get("event_timestamp")
    if isinstance(timestamp, str):
        try:
            event = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            boundary = datetime.strptime(current, "%Y%m%d").replace(tzinfo=UTC)
            if event.tzinfo is None:
                return "unknown"
            if event.date() > boundary.date():
                return "future"
            before = datetime.fromisoformat(cutoff["event_timestamp"].replace("Z", "+00:00"))
            if before.tzinfo is None:
                return "unknown"
            return "after_cutoff" if event > before else "before_cutoff"
        except (ValueError, KeyError, TypeError):
            return "unknown"
    sequence = row.get("event_sequence")
    if (
        type(sequence) is int
        and type(cutoff.get("event_sequence")) is int
        and row.get("sequence_scope")
        and row["sequence_scope"] == cutoff.get("sequence_scope")
    ):
        return "after_cutoff" if sequence > cutoff["event_sequence"] else "before_cutoff"
    return "unknown"


def inspect(
    root: Path,
    producers: list[Path],
    seen: dict[str, str],
    baseline_date: str,
    current_date: str,
    cutoff: dict[str, Any],
) -> dict[str, Any]:
    """Screen one corpus twice while retaining every failed custody operand."""
    rows: list[dict[str, Any]] = []
    inventory = dict(seen)
    conflicts: set[str] = set()
    for producer in producers:
        base: dict[str, Any] = dict(
            producer_path=str(producer),
            producer_sha256=sha256_file(producer) if producer.is_file() else None,
            role="science_producer",
            status="excluded",
            reason=None,
        )
        try:
            document = json.loads(producer.read_text())
        except (ValueError, OSError):
            rows.append(dict(base, status="malformed", reason="malformed_producer"))
            continue
        if not isinstance(document, dict) or not isinstance(
            document.get("source_artifact_hashes"), dict
        ):
            rows.append(dict(base, status="malformed", reason="unsupported_source_hashes"))
            continue
        date = str(document.get("run_date", ""))
        base.update(run_date=date, calendar_eligible=date > baseline_date)
        for label, declared in document["source_artifact_hashes"].items():
            if not isinstance(label, str) or not label.endswith(".json"):
                continue
            raw = Path(label)
            raw = (raw if raw.is_absolute() else root / raw).resolve()
            if not raw.is_relative_to((root / "results/raw").resolve()):
                if label.startswith("results/raw/"):
                    rows.append(dict(base, reason="path_escape", source_path=label))
                continue
            item = dict(
                base,
                source_path=str(raw),
                expected_sha256=declared,
                source_sha256=sha256_file(raw) if raw.is_file() else None,
            )
            expected = declared.get("sha256") if isinstance(declared, dict) else declared
            if not isinstance(expected, str) or not re.fullmatch(
                r"(?:sha256:)?[0-9a-f]{64}", expected
            ):
                rows.append(dict(item, reason="unsupported_source_hash"))
                continue
            expected = expected if expected.startswith("sha256:") else "sha256:" + expected
            item["expected_sha256"] = expected
            if item["source_sha256"] != expected:
                rows.append(
                    dict(item, reason="missing_raw" if not raw.is_file() else "raw_hash_mismatch")
                )
                continue
            if not re.fullmatch(r"\d{8}", date) or date > current_date:
                rows.append(dict(item, reason="future_or_missing_date"))
                continue
            if document.get("flagged_adversarial") is True or document.get("verdict_class") not in {
                "null",
                "positive",
                "circular_positive",
            }:
                rows.append(dict(item, status="disqualified", reason="disqualified_producer"))
                continue
            if "raw:" + expected in seen:
                rows.append(dict(item, reason="seen_raw_receipt"))
                continue
            try:
                episodes = extract_rows(json.loads(raw.read_text()))
            except ValueError:
                rows.append(dict(item, status="malformed", reason="malformed_raw"))
                continue
            for episode in episodes:
                receipt = episode.get("trajectory_supervisor")
                if not isinstance(receipt, dict):
                    continue
                invocation = episode.get("invocation_id", episode.get("attempt"))
                identity = canonical_hash([episode.get("game"), episode.get("seed"), invocation])
                receipt_id = str(episode.get("receipt_id", identity))
                bindings = ["invocation:" + identity, "receipt:" + receipt_id]
                digest = canonical_hash(episode)
                row = dict(
                    item,
                    game=episode.get("game"),
                    seed=episode.get("seed"),
                    invocation_id=invocation,
                    receipt_id=receipt_id,
                    content_sha256=digest,
                    identity_bindings=bindings,
                    solve_provenance=episode.get("solve_provenance"),
                    chronology=event_order(episode, cutoff, current_date),
                )
                reason = None
                if invocation is None or row["game"] is None or row["seed"] is None:
                    reason = "missing_identity"
                elif row["solve_provenance"] != "live_agent_self_discovery":
                    reason = "non_live_provenance"
                elif row["chronology"] == "future":
                    reason = "future_event"
                elif any(key in seen and seen[key] != digest for key in bindings):
                    reason = "changed_receipt"
                elif any(key in inventory and inventory[key] != digest for key in bindings):
                    conflicts.update(bindings)
                    reason = "conflicting_identity"
                elif any(key in inventory for key in bindings):
                    reason = "identical_retry"
                if reason:
                    rows.append(dict(row, reason=reason))
                    continue
                inventory.update({key: digest for key in bindings})
                kind = classify_receipt(episode)
                if kind != "applied" or not receipt["redirects"]:
                    rows.append(
                        dict(
                            row,
                            status="shadow" if kind == "shadow" else "excluded",
                            reason="shadow_control" if kind == "shadow" else "no_redirects",
                        )
                    )
                    continue
                termination = episode.get("termination") or {}
                for index, redirect in enumerate(receipt["redirects"]):
                    if not isinstance(redirect, dict) or redirect.get("arm") not in ARM_ORDER:
                        rows.append(dict(row, status="malformed", reason="malformed_redirect"))
                        continue
                    resolved = redirect.get("resolved_by_levelup")
                    pending = redirect.get(
                        "pending_at_resolution", redirect.get("co_credited_count")
                    )
                    known = type(pending) is int and pending > 0 and resolved is True
                    censored = resolved is False and termination.get("reason") in {
                        "action_limit",
                        "time_limit",
                        "timeout",
                        "collection_cap",
                    }
                    status = "completed" if isinstance(resolved, bool) else "unknown"
                    status = "censored" if censored else status
                    rows.append(
                        dict(
                            row,
                            status=status,
                            reason=termination.get("reason") if censored else None,
                            event_id=canonical_hash(
                                [
                                    identity,
                                    redirect.get("id", redirect.get("action_index", index)),
                                    redirect["arm"],
                                ]
                            ),
                            arm=redirect["arm"],
                            resolved_by_levelup=resolved,
                            actions_to_levelup=redirect.get("actions_to_levelup"),
                            pending_at_resolution=pending,
                            helped_sole=(pending == 1) if known else None,
                            helped_share=(1 / pending) if known else None,
                            termination=termination,
                            prospective=row["chronology"] == "after_cutoff",
                            arms_enabled=receipt.get("arms_enabled"),
                            arms_used=receipt.get("arms_used"),
                            stagnations_unredirected=receipt.get("stagnations_unredirected"),
                        )
                    )
    for row in rows:
        if conflicts.intersection(row.get("identity_bindings", [])):
            row.update(status="excluded", reason="conflicting_identity")
    result = reduce_rows(rows)
    result.update(rows=rows, receipt_inventory=inventory)
    return result


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recount observations without converting recovery into performance benefit."""
    live = [r for r in rows if r["status"] in {"completed", "censored", "unknown"}]
    per_game = {
        game: [r for r in live if str(r["game"]) == game]
        for game in sorted({str(r["game"]) for r in live})
    }
    arms: dict[str, Any] = {}
    decisions = []
    for arm in sorted({r["arm"] for r in live}):
        selected = [r for r in live if r["arm"] == arm]
        closed = [r for r in selected if r["status"] == "completed"]
        helped = sum(r["resolved_by_levelup"] is True for r in closed)
        games = sorted({str(r["game"]) for r in closed})
        lower, upper = wilson_bounds(helped, len(closed))
        held = {}
        for game in games:
            others = [r for r in closed if str(r["game"]) != game]
            hits = sum(r["resolved_by_levelup"] is True for r in others)
            held[game] = dict(
                n=len(others), helped=hits, wilson=list(wilson_bounds(hits, len(others)))
            )
        floor = len(closed) >= 30 and len(games) >= 5
        decision = "unchanged"
        if floor and helped == 0 and upper < 0.10:
            decision = "propose_deprioritization"
        elif floor and lower > 0.20 and all(cell["wilson"][0] > 0.20 for cell in held.values()):
            decision = "propose_raise_priority"
        arms[arm] = dict(
            fired=len(selected),
            uncensored=len(closed),
            censored=sum(r["status"] == "censored" for r in selected),
            unknown=sum(r["status"] == "unknown" for r in selected),
            helped=helped,
            helped_sole=sum(r["helped_sole"] is True for r in closed),
            helped_share=sum(r["helped_share"] for r in closed if r["helped_share"] is not None),
            split_unknown=sum(r["helped_share"] is None for r in closed),
            games=len(games),
            lower95=lower,
            upper95=upper,
            leave_one_game_out=held,
            floor_met=floor,
        )
        decisions.append(
            dict(arm=arm, decision=decision, defaults_changed=False, causal_credit=False)
        )
    statuses = Counter(r["status"] for r in rows)
    exhausted = any(
        isinstance(r.get("arms_enabled"), list)
        and r["arms_enabled"]
        and set(r["arms_enabled"]) <= set(r.get("arms_used") or [])
        and (r.get("stagnations_unredirected") or 0) > 0
        for r in live
    )
    return dict(
        recovered_rows=live,
        per_game_results=per_game,
        arm_outcomes=arms,
        refinement_decisions=decisions,
        identity_filter_count=len(live),
        calendar_filter_count=sum(bool(r["calendar_eligible"]) for r in live),
        chronology_unknown_count=sum(r["chronology"] == "unknown" for r in live),
        disposition_counts=dict(statuses),
        new_level_solves_claimed=0,
        shared_method_gap="exhausted arms without progress"
        if exhausted and not any(r["resolved_by_levelup"] is True for r in live)
        else None,
        sample_size_budget=dict(
            intended=len(rows),
            eligible=len(live),
            started=len(live),
            completed=statuses["completed"],
            failed=statuses["malformed"],
            censored=statuses["censored"],
            excluded=sum(statuses[k] for k in ("excluded", "shadow", "disqualified")),
            unknown=statuses["unknown"],
            independent=len(per_game),
            independent_n=len(per_game),
            invocations=len({(r["game"], r["seed"], r["invocation_id"]) for r in live}),
            unit="authenticated_supervisor_redirect",
            independence_scope="exposed_development",
        ),
    )


def replay(value: dict[str, Any]) -> list[str]:
    """A changed headline must fail cold replay against its primitive rows."""
    return [
        key for key, observed in reduce_rows(value["rows"]).items() if value.get(key) != observed
    ]
