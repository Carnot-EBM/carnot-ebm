"""Cross-run refinement over live trajectory-supervisor outcomes (REQ-ARC-WMTE-6720).

WHY. REQ-ARC-WMTE-6640 made redirects measurable, but each run's receipt
dies with its scratch directory. This module is the unattended between-runs
step: it ingests applied receipts into a durable ledger under ops/, then
applies a frozen evidence contract that RECOMMENDS arm-table changes and
never applies them. A level-up FOLLOWING a redirect is not proof the
redirect caused it, so a human applies changes, or nobody does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone, UTC
from pathlib import Path
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_ORDER,
    MAX_UNREDIRECTED_WINDOWS,
)

LEDGER_SCHEMA = "carnot.arc.supervisor_refinement_ledger.v1"

# REQ-ARC-WMTE-7030: the arms a run could fire before receipts said so. Rows written before
# 2026-09-05 carry no `arms_enabled`. The tool rung was env-gated and default OFF from the day
# it was added (REQ-ARC-WMTE-6760), so a legacy row could fire these three plus any arm it
# actually fired. Measured 2026-09-05: reading `ARM_ORDER` as the exhaustion set hid 4 of the
# 5 exhausted cells (53 of 64 unredirected windows), because only one run had the tool rung on.
LEGACY_DEFAULT_ARMS = (ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY)

# REQ-ARC-WMTE-7031: the per-window state fields the ledger keeps from a receipt row.
UNREDIRECTED_WINDOW_FIELDS = (
    "action_index",
    "level",
    "arms_used",
    "goal_bias_installed",
    "induced",
    "induction_attempts",
    "attempt_cap_reached",
    "new_transitions_since_induction",
    "evidence_floor_met",
    "diversity_active",
)
# The exhaustion summary counts windows where each of these read True.
EXHAUSTION_FLAGS = (
    "goal_bias_installed",
    "induced",
    "attempt_cap_reached",
    "evidence_floor_met",
    "diversity_active",
)
RECOMMENDATION_SCHEMA = "carnot.arc.supervisor_refinement_recommendation.v1"
DEFAULT_LEDGER_PARTS = ("ops", "arc_supervisor_refinement_ledger.json")

# Evidence floor. At fired=10 and helped=0, a true follow rate of 0.25 gives
# zero credits with probability 0.75**10 = 0.056 — the smallest count where
# an all-zero record rejects even a modest rate near the 5 percent level.
# Below this, the tool says "insufficient evidence" instead of ranking noise.
MIN_FIRED_PER_ARM = 10

# One-sided 95 percent normal quantile for Wilson score bounds. Stdlib-only
# on purpose: the unattended step must not depend on scipy being installed.
WILSON_Z = 1.6448536269514722

CAUSAL_CAVEAT = (
    "resolved_by_levelup records that a level-up FOLLOWED a redirect inside "
    "the same progress-free span. One level-up credits every pending "
    "redirect (observed live 2026-08-27: two arms credited by one level-up). "
    "It is not evidence of cause. A human applies these, or nobody does."
)

# REQ-ARC-WMTE-7012: the live-path eval (`scripts/arc_leaderboard_eval.py`) writes
# `{"per_game": [row, ...]}` artifacts into this directory. Until 2026-09-04 this tool
# could read only harness `rows.json` files, so 25 applied redirects across five eval
# rows sat unread while the ledger held 6. A directory scan now takes every `*.json`
# inside a directory with this name, partial files included (a banked game row is
# complete; the final artifact's identical row dedupes by hash).
EVAL_RUNS_DIR_NAME = "arc_leaderboard_eval_runs"
# The eval-run fields this consumer requires (scripts/eval_run_consumer_field_lint.py):
# document-level first (the row container and the context copied down), then row-level.
EVAL_RUN_FIELDS_READ = (
    "per_game",
    "random_seed",
    "policy",
    "budget",
    "trajectory_supervisor",
    "game",
    "levels",
    "actions",
)
# The heartbeat file (REQ-ARC-WMTE-7010) lives in the same directory and is not a record.
PROGRESS_FILE_SUFFIX = ".progress.json"

STATUS_NO_RECEIPTS = "no_receipts_ingested"
STATUS_NO_FIRINGS = "no_firings_nothing_to_refine"
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_RECOMMENDATION = "recommendation_available"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def receipt_id_for_row(row: dict[str, Any]) -> str:
    """Hash the FULL source row, not just the receipt. A byte-identical copy
    dedupes; a genuine re-run differs in measured fields (wall time, frame
    counts) and correctly counts as new evidence."""

    return "sha256:" + hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def classify_receipt(row: dict[str, Any]) -> str:
    """One of: applied | shadow | error | other | absent (REQ-6720 rule 1).

    Only an explicit applied receipt is redirect evidence. Shadow receipts
    hold counterfactuals that never ran; ingesting one would count a
    redirect that was never applied — the field-names-lie class."""

    receipt = row.get("trajectory_supervisor")
    if not isinstance(receipt, dict):
        return "absent"
    if "error" in receipt:
        return "error"
    if receipt.get("mode") == "shadow" or "would_have_redirects" in receipt:
        return "shadow"
    if (
        receipt.get("mode") == "applied"
        and receipt.get("enabled") is True
        and isinstance(receipt.get("redirects"), list)
    ):
        return "applied"
    return "other"


def _evidence_from_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    receipt = row["trajectory_supervisor"]
    redirects: list[dict[str, Any]] = []
    for item in receipt.get("redirects") or []:
        if not isinstance(item, dict) or item.get("arm") is None:
            continue
        co_credited = item.get("co_credited_count")
        redirects.append(
            {
                "arm": str(item.get("arm")),
                "action_index": item.get("action_index"),
                "level": item.get("level"),
                "resolved_by_levelup": item.get("resolved_by_levelup") is True,
                "actions_to_levelup": item.get("actions_to_levelup"),
                # REQ-ARC-WMTE-7013: None on rows written before the field existed,
                # and on rows no level-up ever credited.
                "co_credited_count": co_credited if isinstance(co_credited, int) else None,
            }
        )
    # REQ-ARC-WMTE-7030: None on rows written before the receipt carried the set.
    arms_enabled = receipt.get("arms_enabled")
    arms_enabled = [str(arm) for arm in arms_enabled] if isinstance(arms_enabled, list) else None
    # REQ-ARC-WMTE-7031: the state at each exhausted window, bounded at the receipt's own cap.
    windows: list[dict[str, Any]] = []
    for item in receipt.get("unredirected_windows") or []:
        if isinstance(item, dict) and len(windows) < MAX_UNREDIRECTED_WINDOWS:
            windows.append({k: item.get(k) for k in UNREDIRECTED_WINDOW_FIELDS})
    dropped = receipt.get("unredirected_windows_dropped")
    return {
        "source": source,
        "game": row.get("game"),
        "seed": row.get("seed"),
        "harness_arm": row.get("arm"),
        "window": receipt.get("window"),
        "mode": "applied",
        "actions_observed": receipt.get("actions_observed"),
        "stagnations_unredirected": int(receipt.get("stagnations_unredirected") or 0),
        "levels": row.get("levels"),
        "actions": row.get("actions"),
        "arms_enabled": arms_enabled,
        "unredirected_windows": windows,
        "unredirected_windows_dropped": dropped if isinstance(dropped, int) else None,
        "redirects": redirects,
    }


def _control_from_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    """A shadow receipt as CONTROL evidence (REQ-ARC-WMTE-7032).

    A shadow run applied nothing, so `levelup_followed_without_redirect` on one of its
    would-have rows is the base rate: the same level-up, in the same cell, with no arm
    pulled. This is not redirect evidence and never enters `entries`."""

    receipt = row["trajectory_supervisor"]
    would_have: list[dict[str, Any]] = []
    for item in receipt.get("would_have_redirects") or []:
        if not isinstance(item, dict) or item.get("arm") is None:
            continue
        would_have.append(
            {
                "arm": str(item.get("arm")),
                "action_index": item.get("action_index"),
                "level": item.get("level"),
                "levelup_followed_without_redirect": (
                    item.get("levelup_followed_without_redirect") is True
                ),
                "actions_to_levelup_without_redirect": item.get(
                    "actions_to_levelup_without_redirect"
                ),
            }
        )
    return {
        "source": source,
        "game": row.get("game"),
        "seed": row.get("seed"),
        "harness_arm": row.get("arm"),
        "window": receipt.get("window"),
        "mode": "shadow",
        "levels": row.get("levels"),
        "actions": row.get("actions"),
        "stagnations_unredirected": int(receipt.get("stagnations_unredirected") or 0),
        "would_have_redirects": would_have,
    }


def enabled_arms_for_entry(entry: dict[str, Any]) -> tuple[set[str], str]:
    """The arms this entry's run could fire, and where that answer came from.

    A declared `arms_enabled` wins; a legacy row falls back to the three default-on arms.
    Either way an arm that fired was enabled, so the fired set is unioned in."""

    fired = {redirect["arm"] for redirect in entry.get("redirects", [])}
    declared = entry.get("arms_enabled")
    if isinstance(declared, list) and declared:
        return {str(arm) for arm in declared} | fired, "receipt"
    return set(LEGACY_DEFAULT_ARMS) | fired, "legacy_default"


def _window_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-window rows an entry kept (REQ-ARC-WMTE-7031); empty for a legacy row."""

    return [w for w in entry.get("unredirected_windows") or [] if isinstance(w, dict)]


def _summarise_windows(windows: list[dict[str, Any]], dropped: int) -> dict[str, Any] | str:
    """Per-flag counts over a list of window rows, or `not_recorded` when the list is empty."""

    if not windows:
        return "not_recorded"
    summary: dict[str, Any] = {
        "windows_recorded": len(windows),
        "windows_dropped": int(dropped),
        "levels": sorted({int(w["level"]) for w in windows if w.get("level") is not None}),
        "arms_used_sets": sorted({",".join(w.get("arms_used") or []) for w in windows}),
    }
    for flag in EXHAUSTION_FLAGS:
        summary[flag] = sum(1 for w in windows if w.get(flag) is True)
    return summary


def exhaustion_summary(entry: dict[str, Any]) -> dict[str, Any] | str:
    """Count what the table saw across ALL of an entry's unredirected windows (REQ-ARC-WMTE-7031).

    Returns the string `not_recorded` for a legacy row, so a reader cannot mistake "no
    window rows were kept" for "every flag read False". This is the whole-entry view; a new-arm
    cell summarises only the rows on its own level (REQ-ARC-WMTE-7033)."""

    return _summarise_windows(
        _window_rows(entry), int(entry.get("unredirected_windows_dropped") or 0)
    )


def exhausted_windows_by_level(entry: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Window rows at which every arm the run could fire was already spent ON THAT LEVEL.

    REQ-ARC-WMTE-7033. The supervisor clears its spent-arm set on every level-up
    (`TrajectorySupervisor.observe`), so "every arm used" is a fact about one level, never about
    the run. A window row's `arms_used` is that per-level set as it stood when the table failed
    to answer, so the test is `enabled <= row.arms_used`, row by row. Pooling the arms fired
    across the whole run would call a run exhausted when no single level ever was (the
    2026-09-05 defect: 5 cells, all false)."""

    enabled, _ = enabled_arms_for_entry(entry)
    by_level: dict[int, list[dict[str, Any]]] = {}
    for row in _window_rows(entry):
        used = row.get("arms_used")
        if row.get("level") is None or not isinstance(used, list):
            continue
        if enabled <= {str(arm) for arm in used}:
            by_level.setdefault(int(row["level"]), []).append(row)
    return by_level


def extract_rows(doc: Any) -> list[dict[str, Any]]:
    """Accept every shape a receipt producer has written.

    Harness shapes: a bare list of rows, or an object with a `rows` list. Live-eval
    shape (REQ-ARC-WMTE-7012): an object with a `per_game` list. The eval keeps
    `random_seed`, `policy` and `budget` at the document level, so those are copied
    down onto each row BEFORE hashing; the same game row in a partial and in the
    final artifact then hashes identically and dedupes. Anything else holds no rows.
    """

    if isinstance(doc, list):
        return [row for row in doc if isinstance(row, dict)]
    if isinstance(doc, dict) and isinstance(doc.get("rows"), list):
        return [row for row in doc["rows"] if isinstance(row, dict)]
    if isinstance(doc, dict) and isinstance(doc.get("per_game"), list):
        rows: list[dict[str, Any]] = []
        for raw in doc["per_game"]:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            row.setdefault("seed", doc.get("random_seed"))
            row.setdefault("arm", f"eval:{doc.get('policy')}:budget{doc.get('budget')}")
            rows.append(row)
        return rows
    return []


def _walk_rows_files(root: Path) -> Iterator[Path]:
    """Yield receipt files under root, skipping any directory that holds a
    `.git` entry (REQ-6720 rule 3). A nested repo clone swept by a recursive
    glob once inflated a corpus from 86 rows to 2,212; a worktree marks
    itself with a `.git` FILE, so both forms prune.

    Two file shapes are collected: every `rows.json` (harness), and every
    `*.json` inside a directory named `arc_leaderboard_eval_runs` (live eval,
    REQ-ARC-WMTE-7012)."""

    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirnames or ".git" in filenames:
            dirnames[:] = []
            continue
        if "rows.json" in filenames:
            yield Path(dirpath) / "rows.json"
        if os.path.basename(dirpath) == EVAL_RUNS_DIR_NAME:
            for name in sorted(filenames):
                if (
                    name.endswith(".json")
                    and not name.startswith(".")
                    and not name.endswith(PROGRESS_FILE_SUFFIX)
                ):
                    yield Path(dirpath) / name


def scan_inputs(inputs: Iterable[Path | str]) -> list[Path]:
    """Resolve explicit files plus directory scans into a stable file list.
    A missing input raises: the unattended step must fail loud, never
    silently report clean without reading what it was pointed at."""

    found: set[Path] = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_file():
            found.add(path)
        elif path.is_dir():
            found.update(_walk_rows_files(path))
        else:
            raise FileNotFoundError(f"input does not exist: {path}")
    return sorted(found, key=str)


def empty_ledger() -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "created_at": None,
        "updated_at": None,
        "entries": {},
        # REQ-ARC-WMTE-7032: shadow receipts, kept apart from redirect evidence.
        "controls": {},
        "recommendation": None,
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_ledger()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != LEDGER_SCHEMA:
        raise ValueError(f"unsupported ledger schema in {path}")
    if not isinstance(data.get("entries"), dict):
        raise ValueError(f"malformed ledger entries in {path}")
    # A ledger written before REQ-ARC-WMTE-7032 has no control pool; an empty one is honest.
    if not isinstance(data.get("controls"), dict):
        data["controls"] = {}
    return data


def save_ledger(ledger: dict[str, Any], path: Path) -> None:
    """Atomic replace so a crash mid-write cannot half-destroy the only
    durable copy of the evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def ingest_files(ledger: dict[str, Any], files: Sequence[Path], now_iso: str) -> dict[str, int]:
    counts = {
        "files_read": 0,
        "rows_seen": 0,
        "applied_new": 0,
        "applied_duplicate": 0,
        "shadow_observed": 0,
        "error_rows": 0,
        "other_receipts": 0,
        "rows_without_receipt": 0,
        "controls_new": 0,
        "controls_duplicate": 0,
    }
    entries = ledger["entries"]
    controls = ledger.setdefault("controls", {})
    for file_path in files:
        doc = json.loads(file_path.read_text(encoding="utf-8"))
        counts["files_read"] += 1
        for row in extract_rows(doc):
            counts["rows_seen"] += 1
            kind = classify_receipt(row)
            if kind == "applied":
                receipt_id = receipt_id_for_row(row)
                if receipt_id in entries:
                    counts["applied_duplicate"] += 1
                else:
                    entry = _evidence_from_row(row, str(file_path))
                    entry["receipt_id"] = receipt_id
                    entry["ingested_at"] = now_iso
                    entries[receipt_id] = entry
                    counts["applied_new"] += 1
            elif kind == "shadow":
                counts["shadow_observed"] += 1
                # REQ-ARC-WMTE-7032: a control, never an entry (REQ-6720 rule 1 holds).
                control_id = receipt_id_for_row(row)
                if control_id in controls:
                    counts["controls_duplicate"] += 1
                else:
                    control = _control_from_row(row, str(file_path))
                    control["receipt_id"] = control_id
                    control["ingested_at"] = now_iso
                    controls[control_id] = control
                    counts["controls_new"] += 1
            elif kind == "error":
                counts["error_rows"] += 1
            elif kind == "other":
                counts["other_receipts"] += 1
            else:
                counts["rows_without_receipt"] += 1
    return counts


def wilson_bounds(helped: int, fired: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval. Chosen over a normal approximation because it
    behaves at 0/n and n/n — exactly the counts the retire and raise rules
    read — and needs only math.sqrt."""

    if fired <= 0:
        return (0.0, 1.0)
    phat = helped / fired
    z2 = z * z
    denominator = 1.0 + z2 / fired
    center = phat + z2 / (2.0 * fired)
    margin = z * math.sqrt(phat * (1.0 - phat) / fired + z2 / (4.0 * fired * fired))
    return ((center - margin) / denominator, (center + margin) / denominator)


NOT_DECIDABLE_NO_WINDOW_ROWS = "no_window_rows_recorded"
NOT_DECIDABLE_ROWS_DROPPED = "window_rows_dropped_past_cap"


def _new_arm_cells(
    entries: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One cell per (receipt, level) where the table ran dry with every arm spent ON THAT LEVEL.

    REQ-ARC-WMTE-7033: exhaustion is decided on the axis the arms reset on. A level is exhausted
    when at least one of its window rows shows every enabled arm already used (REQ-7030 for the
    enabled set, REQ-7031 for the rows). A receipt from before the rows existed carries only a
    run-pooled count, so its levels cannot be read; it goes to the second list, `not_decidable`,
    with the reason, instead of being guessed at from arms fired anywhere in the run."""

    cells: list[dict[str, Any]] = []
    not_decidable: list[dict[str, Any]] = []
    for entry in entries:
        stagnations = int(entry.get("stagnations_unredirected") or 0)
        dropped = int(entry.get("unredirected_windows_dropped") or 0)
        windows = _window_rows(entry)
        base = {
            "game": entry.get("game"),
            "seed": entry.get("seed"),
            "window": entry.get("window"),
            "source": entry.get("source"),
            "levels": entry.get("levels"),
            "stagnations_unredirected": stagnations,
        }
        if stagnations > 0 and not windows:
            not_decidable.append({**base, "reason": NOT_DECIDABLE_NO_WINDOW_ROWS})
            continue
        enabled, enabled_source = enabled_arms_for_entry(entry)
        by_level = exhausted_windows_by_level(entry)
        if not by_level:
            if dropped > 0:
                # Rows past the cap were not kept; a kept row would decide, a missing one cannot.
                not_decidable.append(
                    {**base, "reason": NOT_DECIDABLE_ROWS_DROPPED, "windows_dropped": dropped}
                )
            continue
        redirects = [r for r in entry.get("redirects", []) if isinstance(r, dict)]
        for level in sorted(by_level):
            rows = by_level[level]
            on_level = [r for r in redirects if r.get("level") == level]
            # A level-up credits every redirect pending on that level, so "resolved" is the
            # same answer for each of them; read it from any one.
            resolved = any(r.get("resolved_by_levelup") is True for r in on_level)
            levelup_at = next(
                (
                    int(r["action_index"]) + int(r["actions_to_levelup"])
                    for r in on_level
                    if r.get("resolved_by_levelup") is True
                    and isinstance(r.get("action_index"), int)
                    and isinstance(r.get("actions_to_levelup"), int)
                ),
                None,
            )
            first = min(
                (int(w["action_index"]) for w in rows if isinstance(w.get("action_index"), int)),
                default=None,
            )
            cells.append(
                {
                    **base,
                    "level": level,
                    "arms_enabled": sorted(enabled),
                    "arms_enabled_source": enabled_source,
                    "arms_fired_on_level": sorted({str(r.get("arm")) for r in on_level}),
                    "exhausted_windows": len(rows),
                    "windows_on_level": sum(1 for w in windows if w.get("level") == level),
                    "windows_dropped": dropped,
                    "first_exhausted_action_index": first,
                    # True when the level was later cleared with no new arm: the table ran
                    # dry and the classical path got through anyway. Weaker evidence for a
                    # new arm than a level that never resolved.
                    "level_resolved_by_levelup": resolved,
                    "actions_from_first_exhaustion_to_levelup": (
                        levelup_at - first if levelup_at is not None and first is not None else None
                    ),
                    "exhaustion_states": _summarise_windows(rows, dropped),
                }
            )
    return cells, not_decidable


def _control_index(controls: Iterable[dict[str, Any]]) -> set[tuple[Any, ...]]:
    """Cells (game, seed, window, level, arm) where a shadow run leveled up after the point
    the arm would have fired, with nothing applied (REQ-ARC-WMTE-7032)."""

    followed: set[tuple[Any, ...]] = set()
    for control in controls:
        for row in control.get("would_have_redirects") or []:
            if row.get("levelup_followed_without_redirect") is True:
                followed.add(
                    (
                        control.get("game"),
                        control.get("seed"),
                        control.get("window"),
                        row.get("level"),
                        row.get("arm"),
                    )
                )
    return followed


def evaluate(ledger: dict[str, Any], now_iso: str) -> dict[str, Any]:
    """Apply the frozen contract to the whole ledger. Pure function of the
    ledger content so a re-evaluation after source deletion reproduces the
    same answer (SCENARIO-6720-2)."""

    entries = list(ledger["entries"].values())
    controls = list((ledger.get("controls") or {}).values())
    control_followed = _control_index(controls)
    redirects: list[dict[str, Any]] = []
    for entry in entries:
        for redirect in entry.get("redirects", []):
            # REQ-ARC-WMTE-7032: a credit is control-matched when a shadow run in the same
            # (game, seed, window, level) leveled up after the same arm would have fired.
            # Tagged on a copy; the ledger entry itself is not rewritten by evaluation.
            tagged = dict(redirect)
            key = (
                entry.get("game"),
                entry.get("seed"),
                entry.get("window"),
                redirect.get("level"),
                redirect.get("arm"),
            )
            tagged["control_matched"] = bool(
                redirect.get("resolved_by_levelup") and key in control_followed
            )
            redirects.append(tagged)

    arms_seen = sorted({redirect["arm"] for redirect in redirects} - set(ARM_ORDER))
    arm_names = [*ARM_ORDER, *arms_seen]
    per_arm: list[dict[str, Any]] = []
    for arm in arm_names:
        arm_redirects = [redirect for redirect in redirects if redirect["arm"] == arm]
        fired = len(arm_redirects)
        helped = sum(1 for redirect in arm_redirects if redirect["resolved_by_levelup"])
        helped_matched = sum(1 for redirect in arm_redirects if redirect["control_matched"])
        lower, upper = wilson_bounds(helped, fired)
        actions = sorted(
            redirect["actions_to_levelup"]
            for redirect in arm_redirects
            if redirect["resolved_by_levelup"] and redirect["actions_to_levelup"] is not None
        )
        # REQ-ARC-WMTE-7013: the strict companions to `helped`. `helped_sole` counts
        # credits this arm did not share with another pending redirect; `helped_share`
        # splits each level-up evenly over the redirects it credited. Rows written
        # before the field existed carry None and count in neither.
        helped_sole = 0
        helped_share = 0.0
        helped_known = 0
        for redirect in arm_redirects:
            k = redirect.get("co_credited_count")
            if redirect["resolved_by_levelup"] and isinstance(k, int) and k > 0:
                helped_known += 1
                helped_share += 1.0 / k
                if k == 1:
                    helped_sole += 1
        sole_lower, sole_upper = wilson_bounds(helped_sole, fired)
        per_arm.append(
            {
                "arm": arm,
                "fired": fired,
                "helped": helped,
                "help_follow_rate": round(helped / fired, 6) if fired else None,
                "wilson_lower": round(lower, 6),
                "wilson_upper": round(upper, 6),
                "helped_sole": helped_sole,
                "helped_share": round(helped_share, 4),
                "helped_with_known_split": helped_known,
                "sole_wilson_lower": round(sole_lower, 6),
                "sole_wilson_upper": round(sole_upper, 6),
                # REQ-ARC-WMTE-7032: credits a shadow control reproduced with nothing applied,
                # and the remainder. The frozen rules keep reading pooled `helped`.
                "helped_matched_by_control": helped_matched,
                "helped_beyond_control": helped - helped_matched,
                "actions_to_levelup": actions,
                "meets_floor": fired >= MIN_FIRED_PER_ARM,
                "floor_shortfall": max(0, MIN_FIRED_PER_ARM - fired),
            }
        )

    recommendations: list[dict[str, Any]] = []
    for arm_row in per_arm:
        fired = arm_row["fired"]
        helped = arm_row["helped"]
        if fired >= MIN_FIRED_PER_ARM and helped == 0:
            recommendations.append(
                {
                    "kind": "retire_candidate",
                    "arm": arm_row["arm"],
                    "evidence": {
                        "fired": fired,
                        "helped": helped,
                        "wilson_upper": arm_row["wilson_upper"],
                    },
                    "why": (
                        f"{fired} firings and zero follow-ups under a metric that "
                        "shares credit generously; the help-follow rate is below "
                        f"{arm_row['wilson_upper']} at one-sided 95 percent."
                    ),
                }
            )
            continue
        others = [row for row in per_arm if row["arm"] != arm_row["arm"]]
        others_fired = sum(row["fired"] for row in others)
        others_helped = sum(row["helped"] for row in others)
        _, others_upper = wilson_bounds(others_helped, others_fired)
        if (
            fired >= MIN_FIRED_PER_ARM
            and others_fired >= MIN_FIRED_PER_ARM
            and arm_row["wilson_lower"] > others_upper
        ):
            recommendations.append(
                {
                    "kind": "raise_priority_candidate",
                    "arm": arm_row["arm"],
                    "evidence": {
                        "fired": fired,
                        "helped": helped,
                        "wilson_lower": arm_row["wilson_lower"],
                        "others_fired": others_fired,
                        "others_helped": others_helped,
                        "others_wilson_upper": round(others_upper, 6),
                    },
                    "why": (
                        "this arm's lower bound exceeds the pooled other arms' "
                        "upper bound under the same post-hoc crediting bias."
                    ),
                }
            )

    cells, exhaustion_not_decidable = _new_arm_cells(entries)
    new_arm_specification: dict[str, Any] | None = None
    if cells:
        new_arm_specification = {
            "audience": "human",
            "instruction": (
                "Propose ONE new curated arm for ARM_ORDER in "
                "python/carnot/agentic/arc_trajectory_supervisor.py. Arm growth "
                "stays human on a 27B generator; this tool never generates an "
                "arm implementation."
            ),
            "trigger": (
                "on ONE level, every arm the run could fire (arms_enabled) was already "
                "spent when a stagnation window passed with no arm to fire; one cell per "
                "(receipt, level) below (REQ-ARC-WMTE-7033)"
            ),
            "cells": cells,
        }

    if not entries:
        status = STATUS_NO_RECEIPTS
    elif not redirects:
        status = STATUS_NO_FIRINGS
    elif recommendations or new_arm_specification:
        status = STATUS_RECOMMENDATION
    else:
        status = STATUS_INSUFFICIENT

    return {
        "schema": RECOMMENDATION_SCHEMA,
        "generated_at": now_iso,
        "status": status,
        "contract": {
            "min_fired_per_arm": MIN_FIRED_PER_ARM,
            "wilson_z": WILSON_Z,
            "rules": [
                "retire_candidate: fired >= floor and helped == 0",
                "raise_priority_candidate: arm and pooled others both at floor, "
                "arm lower bound > others upper bound",
                "new_arm_specification: on one level, a window row shows every arm the run "
                "could fire (arms_enabled) already spent on that level; a receipt with no "
                "window rows is listed as not decidable, never as a cell",
            ],
        },
        "recommendation_only": True,
        "causal_caveat": CAUSAL_CAVEAT,
        "evidence": {
            "receipts": len(entries),
            "redirects": len(redirects),
            "games": sorted({str(entry.get("game")) for entry in entries}),
            "stagnations_unredirected_total": sum(
                int(entry.get("stagnations_unredirected") or 0) for entry in entries
            ),
            # REQ-ARC-WMTE-7032: how much of the pooled credit a control reproduced.
            "controls": len(controls),
            "helped_total": sum(1 for r in redirects if r["resolved_by_levelup"]),
            "helped_matched_by_control_total": sum(1 for r in redirects if r["control_matched"]),
            # REQ-ARC-WMTE-7033: receipts that stagnated but carry no per-level window rows.
            "exhaustion_not_decidable": len(exhaustion_not_decidable),
        },
        "per_arm": per_arm,
        "recommendations": recommendations,
        "new_arm_specification": new_arm_specification,
        "exhaustion_not_decidable": exhaustion_not_decidable,
    }


def render_report(recommendation: dict[str, Any]) -> str:
    """Human report. The insufficient case is deliberately LOUD: a quiet
    weak ranking is the churn mode this tool exists to refuse."""

    lines: list[str] = ["ARC supervisor refinement (REQ-ARC-WMTE-6720)"]
    status = recommendation["status"]
    if status == STATUS_NO_RECEIPTS:
        lines.append(
            "NO RECEIPTS INGESTED — nothing to refine. This honest empty "
            "report satisfies the generalization-floor slot."
        )
    elif status == STATUS_NO_FIRINGS:
        lines.append(
            "NO FIRINGS — the supervisor never redirected. Nothing to "
            "refine; this report satisfies the generalization-floor slot."
        )
    elif status == STATUS_INSUFFICIENT:
        lines.append(
            "INSUFFICIENT EVIDENCE — firings exist but no rule crosses its "
            f"floor (min {MIN_FIRED_PER_ARM} firings per arm). No ranking is "
            "emitted; a ranking at these counts would be noise."
        )
    else:
        lines.append("RECOMMENDATION AVAILABLE — recommendation only; a human applies.")
    evidence = recommendation["evidence"]
    lines.append(
        f"evidence: {evidence['receipts']} receipts, {evidence['redirects']} "
        f"redirects, games={','.join(evidence['games']) or 'none'}"
    )
    # REQ-ARC-WMTE-7032: say how much of the pooled credit a control reproduced. A credit
    # the control reproduces is base rate, not an arm effect; a reader must see the split.
    lines.append(
        f"controls: {evidence.get('controls', 0)} shadow receipts; "
        f"{evidence.get('helped_matched_by_control_total', 0)} of "
        f"{evidence.get('helped_total', 0)} credits matched by a control "
        "(same level-up followed with nothing applied)"
    )
    for row in recommendation["per_arm"]:
        lines.append(
            f"  {row['arm']}: fired={row['fired']} helped={row['helped']} "
            f"wilson=[{row['wilson_lower']}, {row['wilson_upper']}] "
            f"sole={row.get('helped_sole', 0)} share={row.get('helped_share', 0.0)} "
            f"beyond_control={row.get('helped_beyond_control', row['helped'])} "
            f"floor_shortfall={row['floor_shortfall']}"
        )
    for item in recommendation["recommendations"]:
        lines.append(f"  RECOMMEND {item['kind']} arm={item['arm']}: {item['why']}")
    spec = recommendation.get("new_arm_specification")
    if spec:
        lines.append(
            "  NEW ARM SPECIFICATION (for a human): on one level every enabled arm was "
            f"spent and stagnation continued, in {len(spec['cells'])} (receipt, level) cell(s):"
        )
        for cell in spec["cells"]:
            # REQ-ARC-WMTE-7033: the cell names its level and whether that level was later
            # cleared anyway, so the reader can weigh the cell before proposing an arm.
            lines.append(
                f"    game={cell['game']} seed={cell['seed']} "
                f"window={cell['window']} level={cell.get('level')} "
                f"exhausted_windows={cell.get('exhausted_windows')} "
                f"level_resolved_by_levelup={cell.get('level_resolved_by_levelup')} "
                f"actions_from_first_exhaustion_to_levelup="
                f"{cell.get('actions_from_first_exhaustion_to_levelup')} "
                f"arms_enabled={','.join(cell.get('arms_enabled') or [])} "
                f"({cell.get('arms_enabled_source', '?')})"
            )
            states = cell.get("exhaustion_states")
            if isinstance(states, dict):
                # REQ-ARC-WMTE-7031: the state the table saw, so the human reads WHY the
                # remaining rungs were ineligible instead of only how many windows passed.
                lines.append(
                    f"      states: windows={states['windows_recorded']} "
                    f"levels={states['levels']} "
                    f"attempt_cap_reached={states['attempt_cap_reached']} "
                    f"diversity_active={states['diversity_active']} "
                    f"goal_bias_installed={states['goal_bias_installed']} "
                    f"induced={states['induced']} "
                    f"evidence_floor_met={states['evidence_floor_met']}"
                )
            else:
                lines.append(f"      states: {states}")
    undecided = recommendation.get("exhaustion_not_decidable") or []
    if undecided:
        # REQ-ARC-WMTE-7033: say out loud which stagnating receipts the reader cannot judge,
        # so zero cells is never mistaken for "no level ever ran dry".
        lines.append(
            f"  EXHAUSTION NOT DECIDABLE for {len(undecided)} receipt(s) that stagnated but "
            "carry no per-level window rows (written before REQ-ARC-WMTE-7031):"
        )
        for item in undecided:
            lines.append(
                f"    game={item['game']} seed={item['seed']} window={item['window']} "
                f"stagnations_unredirected={item['stagnations_unredirected']} "
                f"reason={item['reason']}"
            )
    lines.append(f"caveat: {recommendation['causal_caveat']}")
    return "\n".join(lines)


def _default_ledger_path() -> Path:
    from carnot.paths import repo_path

    return repo_path(*DEFAULT_LEDGER_PARTS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unattended cross-run supervisor refinement (REQ-ARC-WMTE-6720). "
            "Ingests applied trajectory-supervisor receipts from rows.json "
            "files or scan directories into a durable ledger, then prints a "
            "recommendation. Never mutates the arm table."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "rows.json files, live-eval artifacts, or directories scanned for rows.json and "
            "for *.json inside arc_leaderboard_eval_runs (nested repo clones pruned)"
        ),
    )
    parser.add_argument("--ledger", type=Path, default=None, help="ledger path override")
    parser.add_argument("--json", action="store_true", help="print the recommendation as JSON")
    args = parser.parse_args(argv)

    ledger_path = args.ledger if args.ledger is not None else _default_ledger_path()
    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        ledger = load_ledger(ledger_path)
        files = scan_inputs(args.inputs)
        counts = ingest_files(ledger, files, now_iso)
        recommendation = evaluate(ledger, now_iso)
        recommendation["ingest_counts"] = counts
        ledger["recommendation"] = recommendation
        if ledger["created_at"] is None:
            ledger["created_at"] = now_iso
        ledger["updated_at"] = now_iso
        save_ledger(ledger, ledger_path)
    except (OSError, ValueError) as exc:
        # Fail loud (REQ-6720 rule 6): an unattended step that swallows an
        # unreadable input would report clean without having looked.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(recommendation, indent=2, sort_keys=True))
    else:
        print(render_report(recommendation))
    print(f"ledger: {ledger_path}")
    return 0
