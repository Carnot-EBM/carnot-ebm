"""Reduce scored public ARC episodes for REQ-REPORT-7818."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from arc_agi.scorecard import EnvironmentScoreCalculator


FROZEN_MANIFEST_SHA256 = "c2528bdfd84b2f45bbca05b7c6efb368c06b9225dbe2ddc9eac98c1580cb7bfc"


def sha256(path: Path) -> str:
    """Name evidence by completed bytes so a later edit is detectable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_plan(path: Path) -> list[dict[str, Any]]:
    """Allow only the prospective exact child list, including its final readers."""
    if sha256(path) != FROZEN_MANIFEST_SHA256:
        raise ValueError("manifest_mutation")
    data = json.loads(path.read_text())
    commands = data["commands"]
    if len(commands) != 66 or len({row["name"] for row in commands}) != 66:
        raise ValueError("undeclared_child")
    if [row["name"] for row in commands if row["classification"] == "required"] != data[
        "required_checks"
    ]:
        raise ValueError("required_classification_changed")
    return [{key: row[key] for key in ("name", "argv", "classification")} for row in commands]


def seal_log(parent: Path, name: str, attempt: int, data: bytes) -> Path:
    """Write one closed child log once; a retry gets a distinct attempt name."""
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    path = parent / f"{attempt:04d}_{name}_{hashlib.sha256(data).hexdigest()}.log"
    with path.open("xb") as stream:
        stream.write(data)
    return path


def verify_log(path: Path, expected: str) -> bool:
    """A cold reader checks the durable path rather than the live temp log."""
    return path.is_file() and sha256(path) == expected


def score_episode(actions: Sequence[Mapping[str, Any]], baseline: Sequence[int]) -> dict[str, Any]:
    """Apply the installed SDK level formula to both RESET interpretations.

    Each first observed level boundary closes one level. A RESET can be charged
    to the current level or omitted until gateway parity is established.
    """
    boundaries: list[int] = []
    peak = 0
    for index, action in enumerate(actions, 1):
        level = int(action["actual_observation"]["level"])
        while level > peak:
            boundaries.append(index)
            peak += 1
    resets = [index for index, action in enumerate(actions, 1) if action["action"] == "RESET"]
    result: dict[str, Any] = {
        "first_level_up_actions": boundaries[0] if boundaries else None,
        "total_actions": len(actions),
        "reset_count": len(resets),
        "peak_level": peak,
        "human_baseline_actions": list(baseline),
        "level_boundary_actions": boundaries,
    }
    for label, charge in (("charged", True), ("uncharged", False)):
        calc = EnvironmentScoreCalculator()
        prev = 0
        level_actions = []
        for level, human in enumerate(baseline, 1):
            end = boundaries[level - 1] if level <= len(boundaries) else len(actions)
            count = end - prev - (0 if charge else sum(prev < reset <= end for reset in resets))
            level_actions.append(count)
            calc.add_level(level, level <= len(boundaries), count, int(human))
            prev = end
        result[f"level_actions_{label}"] = level_actions
        result[f"score_{label}"] = calc.to_score().score
    return result


def _paired_test(differences: Sequence[float]) -> dict[str, Any]:
    """Enumerate the 256 game-sign assignments and a paired t lower bound."""
    mean = sum(differences) / 8
    null = [
        sum(sign * value for sign, value in zip(signs, differences)) / 8
        for signs in itertools.product((-1, 1), repeat=8)
    ]
    p = sum(value >= mean - 1e-12 for value in null) / 256
    sd = math.sqrt(sum((value - mean) ** 2 for value in differences) / 7)
    return {
        "mean_difference": mean,
        "lower95": mean - 2.365 * sd / math.sqrt(8),
        "upper95": mean + 2.365 * sd / math.sqrt(8),
        "p_one_sided": p,
        "sign_assignments": 256,
    }


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Treat games as independent and suppress broad benefit on missing pairs."""
    games = sorted({str(row["game"]) for row in rows})
    complete = []
    for game in games:
        cells = {(row["seed"], row["arm"]): row for row in rows if row["game"] == game}
        if len(cells) == 6 and all(row.get("status") == "completed" for row in cells.values()):
            complete.append((game, cells))
    result: dict[str, Any] = {
        "independent_n": len(complete),
        "complete_pairs": len(complete) == 8,
        "organic_benefit_score": 0,
        "tests": {},
        "lost_baseline_winning_seed": False,
        "shared_win_action_regression": None,
    }
    if len(complete) != 8:
        return result
    p_rows = []
    shared = []
    for convention in ("charged", "uncharged"):
        for control in ("off", "total"):
            diffs = []
            for _, cells in complete:
                diffs.append(
                    sum(
                        float(cells[seed, "organic"][f"score_{convention}"])
                        - float(cells[seed, control][f"score_{convention}"])
                        for seed in (67815, 67816)
                    )
                    / 2
                )
                for seed in (67815, 67816):
                    organic = cells[seed, "organic"]
                    baseline = cells[seed, control]
                    if float(baseline[f"score_{convention}"]) > 0:
                        if float(organic[f"score_{convention}"]) == 0:
                            result["lost_baseline_winning_seed"] = True
                        elif convention == "charged":
                            shared.append(
                                (int(organic["total_actions"]), int(baseline["total_actions"]))
                            )
            name = f"organic_vs_{control}_{convention}"
            result["tests"][name] = _paired_test(diffs)
            p_rows.append((name, result["tests"][name]["p_one_sided"]))
    for rank, (name, p) in enumerate(sorted(p_rows, key=lambda row: row[1])):
        result["tests"][name]["p_holm"] = min(
            1.0,
            max(
                p_rows_value[1] * (4 - index)
                for index, p_rows_value in enumerate(
                    sorted(p_rows, key=lambda row: row[1])[: rank + 1]
                )
            ),
        )
    result["shared_win_action_regression"] = (
        sum(organic for organic, _ in shared) / sum(control for _, control in shared) - 1
        if shared and sum(control for _, control in shared)
        else None
    )
    result["organic_benefit_score"] = int(
        all(test["lower95"] > 0 and test["p_holm"] <= 0.05 for test in result["tests"].values())
        and not result["lost_baseline_winning_seed"]
        and result["shared_win_action_regression"] is not None
        and result["shared_win_action_regression"] <= 0.10
    )
    return result
