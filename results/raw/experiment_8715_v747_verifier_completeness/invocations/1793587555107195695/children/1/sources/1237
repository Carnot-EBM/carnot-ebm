"""Experiment 4842: ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4842,
SCENARIO-ARC-WMTE-4842-ROTATION-TARGET,
SCENARIO-ARC-WMTE-4842-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4842-STABLE-ARTIFACT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from carnot import experiment_4832_levelup_attempt as previous


REPO = previous.REPO
RESULTS = previous.RESULTS
REGISTRY = previous.REGISTRY
ARTIFACT = RESULTS / "experiment_4842_levelup_attempt.json"

EXPERIMENT = "experiment_4842_levelup_attempt"
SCHEMA = "carnot.exp4842.levelup_attempt.v1"
RESULT_RELATIVE_PATH = "results/experiment_4842_levelup_attempt.json"
REGISTRY_RELATIVE_PATH = previous.REGISTRY_RELATIVE_PATH
RANDOM_SEED = 4842
PUBLIC_FIRST_CONTACT_TARGETS = ("sb26", "lf52", "bp35")
SOLVE_PROVENANCE = previous.SOLVE_PROVENANCE
INFERENCE_SUBSTRATE = previous.INFERENCE_SUBSTRATE

SPEC_REFS = [
    "REQ-ARC-WMTE-4842",
    "SCENARIO-ARC-WMTE-4842-ROTATION-TARGET",
    "SCENARIO-ARC-WMTE-4842-REPRODUCTION-GATE",
    "SCENARIO-ARC-WMTE-4842-STABLE-ARTIFACT",
]

FIELD_PRINCIPLES = dict(previous.FIELD_PRINCIPLES)
REQUIRED_FIELDS = previous.REQUIRED_FIELDS

stable_checksum = previous.stable_checksum
registry_levels = previous.registry_levels
registry_total_levels = previous.registry_total_levels
load_registry = previous.load_registry
summarize_loop_attempt = previous.summarize_loop_attempt
collect_attempt = previous.collect_attempt
collect_attempts = previous.collect_attempts
check_preconditions = previous.check_preconditions


def _checksum_is_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _adaptered_games() -> set[str]:  # pragma: no cover - live adapter registry boundary.
    return previous._adaptered_games()


def _recommend_approach(game: str) -> dict[str, Any]:  # pragma: no cover - live registry boundary.
    return previous._recommend_approach(game)


def _public_rotation(levels: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for game in PUBLIC_FIRST_CONTACT_TARGETS:
        prior = int(levels.get(game, 0))
        rows.append(
            {
                "game": game,
                "known": game in levels,
                "prior_level": prior,
                "status": "unreproduced" if prior < 1 else "already_reproduced",
            }
        )
    return rows


def _candidate_from_public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "game": str(row["game"]),
        "prior_level": int(row.get("prior_level") or 0),
        "target_level": 1,
        "reason": "preferred_public_first_contact",
    }


def _shallowest_solved_rows(
    levels: dict[str, int], adaptered_games: set[str]
) -> list[dict[str, Any]]:
    candidates = [
        (int(level), game)
        for game, level in levels.items()
        if game in adaptered_games and int(level) > 0
    ]
    return [
        {
            "game": game,
            "prior_level": prior,
            "target_level": prior + 1,
            "reason": "shallowest_already_solved_deepen",
        }
        for prior, game in sorted(candidates)
    ]


def select_rotation_target(
    registry: dict[str, Any],
    adaptered_games: set[str] | None = None,
    approach_recommendation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    levels = registry_levels(registry)
    adaptered = _adaptered_games() if adaptered_games is None else set(adaptered_games)
    public_rows = _public_rotation(levels)
    first_contact = [
        _candidate_from_public_row(row)
        for row in public_rows
        if bool(row["known"]) and int(row["prior_level"]) < 1
    ]
    deepen_rows = _shallowest_solved_rows(levels, adaptered)

    if first_contact:
        selected = first_contact[0]
        rotate_after = first_contact[1:] + deepen_rows
    elif deepen_rows:
        selected = deepen_rows[0]
        rotate_after = [row for row in deepen_rows[1:] if row["game"] != selected["game"]]
    else:
        return {
            "game": "none",
            "prior_level": 0,
            "target_level": 0,
            "reason": "no_reproduced_standing_loop_target",
            "public_rotation": public_rows,
            "rotate_if_no_bank": [],
            "shallowest_solved_candidates": [],
            "approach_recommendation": {},
        }

    return {
        **selected,
        "public_rotation": public_rows,
        "rotate_if_no_bank": rotate_after,
        "shallowest_solved_candidates": deepen_rows,
        "approach_recommendation": dict(approach_recommendation or {}),
    }


def build_artifact(
    *,
    registry: dict[str, Any],
    selection: dict[str, Any],
    attempts: list[dict[str, Any]],
    preconditions_checked: dict[str, Any],
) -> dict[str, Any]:
    artifact = previous.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=preconditions_checked,
    )
    artifact.update(
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "spec_refs": SPEC_REFS,
            "field_principles": FIELD_PRINCIPLES,
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "schema_errors": [],
        }
    )
    artifact["reproducibility_checksum"] = stable_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in payload]
    principles = payload.get("field_principles")
    errors.extend(
        f"missing_principle:{field}"
        for field, principle in FIELD_PRINCIPLES.items()
        if not isinstance(principles, dict) or principles.get(field) != principle
    )
    verdict = str(payload.get("honest_verdict") or "")
    checksum = payload.get("reproducibility_checksum")
    checksum_error = (
        "invalid_reproducibility_checksum"
        if not _checksum_is_hex(checksum)
        else "checksum_mismatch"
        if checksum != stable_checksum(dict(payload))
        else ""
    )
    checks = [
        ("honest_verdict_missing_terminal_prefix", not verdict.startswith(("success_", "complete_", "blocked_"))),
        ("solve_provenance_mismatch", payload.get("solve_provenance") != SOLVE_PROVENANCE),
        ("inference_substrate_mismatch", payload.get("inference_substrate") != INFERENCE_SUBSTRATE),
        ("verifier_is_oracle_must_be_true", payload.get("verifier_is_oracle") is not True),
        ("bank_without_offline_reproduction", int(payload.get("new_levels_banked") or 0) > 0 and payload.get("offline_reproduced") is not True),
        ("offline_reproduced_true_without_new_bank", int(payload.get("new_levels_banked") or 0) == 0 and payload.get("offline_reproduced") is True),
        ("retire_if_same_verdict_must_be_true", payload.get("retire_if_same_verdict") is not True),
        ("experiment_mismatch", payload.get("experiment") != EXPERIMENT),
        ("schema_mismatch", payload.get("schema") != SCHEMA),
        ("spec_refs_mismatch", payload.get("spec_refs") != SPEC_REFS),
    ]
    errors.extend([checksum_error] if checksum_error else [])
    errors.extend(name for name, failed in checks if failed)
    return errors


def write_artifact(payload: dict[str, Any], path: Path | None = None) -> Path:
    output = ARTIFACT if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - live CLI boundary.
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)

    registry = load_registry(REGISTRY)
    base_selection = select_rotation_target(registry)
    recommendation = (
        _recommend_approach(str(base_selection["game"]))
        if base_selection.get("game") != "none"
        else {}
    )
    selection = select_rotation_target(
        registry,
        approach_recommendation=recommendation,
    )
    preconditions = check_preconditions(selection)
    attempts = collect_attempts(selection)
    artifact = build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=preconditions,
    )
    write_artifact(artifact, ARTIFACT)
    print(f"target_game={artifact['target_game']}")
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"new_levels_banked={artifact['new_levels_banked']}")
    print(f"schema_errors={artifact['schema_errors']}")
    return 0 if not artifact["schema_errors"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
