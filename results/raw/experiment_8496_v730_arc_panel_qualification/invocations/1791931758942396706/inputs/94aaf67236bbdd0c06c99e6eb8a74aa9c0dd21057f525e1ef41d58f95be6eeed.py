"""REQ-VERIFY-8439: sealed runtime operands prevent invented supervisor outcomes.

The original validated terminal remains useful engineering evidence even when a
later primary flag differs. No current model inference or game execution occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.arc_supervisor_live_panel_8384 import reduce_episode
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
UPSTREAM = "results/experiment_8412_v724_arc_panel_qualification.json"
CANDIDATE = "results/raw/experiment_8412_v724_arc_panel_qualification/terminal_candidate.json"
CANDIDATE_PIN = "sha256:a10cabfbb30c863c835c034c72fa1af7aa1881bdc61d2eb7e1ca16808989eafd"
ARMS = [
    "drop_goal_bias",
    "allow_reinduction",
    "force_exploration_diversity",
    "tool_loop_reinduction",
]


def read(ref: Json) -> Json:
    """A digest supplied by an independent producer is checked before JSON parsing."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("operand_hash")
    return dict(json.loads(path.read_bytes()))


def producer(root: Path, pin: str = CANDIDATE_PIN) -> Json:
    """The preserved publisher candidate binds the validated engineering receipts."""
    value = read(dict(path=str(root / CANDIDATE), sha256=pin))
    side = root / Path(CANDIDATE).parent / "validators" / (pin[7:] + ".json")
    bound = json.loads(side.read_bytes())
    if not (
        bound["primary_sha256"] == pin
        and bound["primary_path"] == str((root / UPSTREAM).absolute())
        and bound["report"]["passed"]
        and value["arc_panel_ready_score"] == 1
        and value["required_checks_passed"]
        and value["flagged_adversarial"] is False
        and not value["adversarial_findings"]
        and value["adapter_withheld"]
        and value["independent_count"] == 2
    ):
        raise ValueError("unqualified_terminal")
    read(value["work_reference"])
    return value


def reduce(source: Json, operands: list[Json]) -> list[Json]:
    """Original assertions and independent hashes bind every episode and arm."""
    units = source["frozen_panel"]["units"]
    if len(units) != 8 or len(operands) != 8 or len(source["rows"]) != 8:
        raise ValueError("episode_roster")
    rows = []
    for index, (unit, operand, original) in enumerate(
        zip(units, operands, source["rows"], strict=True)
    ):
        if operand["unit"] != unit or original["episode_id"] != unit["episode_id"]:
            raise ValueError("episode_assignment")
        for name in ["steps", "terminal"]:
            old_path = Path(original["primitive_directory"]) / (
                "steps.jsonl" if name == "steps" else "episode.json"
            )
            ref = operand[name]
            if (
                ref["sha256"] != source["raw_shard_hashes"][str(old_path)]
                or sha256_file(Path(ref["path"])) != ref["sha256"]
            ):
                raise ValueError("primitive_seal")
        terminal = read(operand["terminal"])
        if terminal["unit"] != unit or not terminal["adapter_withheld"]:
            raise ValueError("terminal_assignment")
        row = reduce_episode(unit, terminal, Path(operand["steps"]["path"]).parent)
        if {k: v for k, v in row.items() if k != "primitive_directory"} != {
            k: v for k, v in original.items() if k != "primitive_directory"
        }:
            raise ValueError("summary_drift")
        rows.append(row)
        print(
            f"[exp8439] phase=primitive_replay completed={index + 1} pending={7 - index}",
            flush=True,
        )
    return rows


def summarize(rows: list[Json]) -> Json:
    """Right-censored events do not estimate failure rates or establish transfer."""
    events = [
        dict(
            event,
            episode_id=row["episode_id"],
            game=row["game"],
            policy_arm=row["arm"],
            fired=1,
            helped=int(event["resolved_by_levelup"]),
        )
        for row in rows
        for event in row["supervisor_outcome_rows"]
    ]
    games = sorted({row["game"] for row in rows})
    cells = []
    for game in games:
        for policy in ["off", "on"]:
            for arm in ARMS:
                selected = [
                    event
                    for event in events
                    if (event["game"], event["policy_arm"], event["arm"]) == (game, policy, arm)
                ]
                resolved = sum(event["resolved_by_levelup"] for event in selected)
                cells.append(
                    dict(
                        game=game,
                        policy_arm=policy,
                        supervisor_arm=arm,
                        fired=len(selected),
                        helped=resolved,
                        resolved_by_levelup=resolved,
                        actions_to_levelup=[event["actions_to_levelup"] for event in selected],
                        pending_censored=len(selected) - resolved,
                        missing_reason="no_firings" if not selected else None,
                    )
                )
    overlap = {}
    for arm in ARMS:
        shared = [
            game
            for game in games
            if all(
                any(
                    c["game"] == game
                    and c["policy_arm"] == policy
                    and c["supervisor_arm"] == arm
                    and c["fired"] > 0
                    for c in cells
                )
                for policy in ["off", "on"]
            )
        ]
        selected = [c for c in cells if c["game"] in shared and c["supervisor_arm"] == arm]
        overlap[arm] = dict(
            games=shared,
            overlapping_games=len(shared),
            required_games=3,
            missing_overlapping_games=max(0, 3 - len(shared)),
            required_resolved_per_cell=5,
            cells=selected,
            missing_resolved_by_cell={
                c["game"] + ":" + c["policy_arm"]: max(0, 5 - c["resolved_by_levelup"])
                for c in selected
            },
            supported=len(shared) >= 3 and all(c["resolved_by_levelup"] >= 5 for c in selected),
        )
    return dict(
        supervisor_outcome_rows=events,
        per_game_results=cells,
        overlap_by_arm=overlap,
        arm_refinement_authorized=False,
        censoring_counts=dict(
            episodes=sum(bool(r.get("censoring")) for r in rows),
            pending_redirects=sum(not e["resolved_by_levelup"] for e in events),
            resolved_redirects=sum(e["resolved_by_levelup"] for e in events),
            applied_redirects=sum(e["applied"] for e in events),
            shadow_redirects=sum(not e["applied"] for e in events),
        ),
        proposed_arm_change=None,
        support_status="insufficient_support",
        summary_comparison=dict(
            passed=True,
            episode_counts=len(rows),
            fired=len(events),
            helped=sum(e["helped"] for e in events),
        ),
    )
