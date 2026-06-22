"""Experiment 4583: shipped exploration-diversity floor transfer.

Spec refs: REQ-CAPSTONE-4583, SCENARIO-CAPSTONE-4583,
SCENARIO-CAPSTONE-4583-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4550_honest_sprint_metric as exp4550


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]

RESULT_RELATIVE_PATH = "results/experiment_4583_diversity_floor_transfer.json"
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
EXPERIMENT_ID = "experiment_4583_diversity_floor_transfer"
SCHEMA = "carnot.exp4583.diversity_floor_transfer.v1"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4583
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "firstwin_count_diversity_on",
    "firstwin_count_diversity_off",
    "firstwin_delta",
    "median_actions_to_first_levelup_with_diversity",
    "median_actions_to_first_levelup_without_diversity",
    "actions_delta",
    "solve_rate_with_diversity",
    "solve_rate_without_diversity",
    "diversity_off_control_passed",
    "false_negative_risk_checked",
    "null_delta_methodology_note",
    "offline_reproduced",
    "chosen_submitted_config",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: diversity_floor_transfer_firstwin_up_<n> OR "
            "complete: diversity_floor_no_transfer_honest_null_gap_sharpened."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline variant exploration, "
            "no LLM load (1s floor)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- diversity-on-stall is a generation-broadening explorer "
            "lever, oracle-DISTINCT from the win-check."
        )
    },
    "firstwin_count_diversity_on": {
        "principle": (
            "the HEADLINE -- held-out first-win count WITH the diversity floor."
        )
    },
    "firstwin_count_diversity_off": {
        "principle": "the matched diversity-OFF baseline on the SAME variants."
    },
    "firstwin_delta": {
        "principle": (
            "on - off (positive = more wins reached = transfer), emitted explicitly "
            "so a null (0) is annotated."
        )
    },
    "median_actions_to_first_levelup_with_diversity": {
        "principle": (
            "ACTION cost WITH diversity -- the leaderboard tiebreaker; diversity must "
            "not blow up actions while reaching more wins."
        )
    },
    "median_actions_to_first_levelup_without_diversity": {
        "principle": "matched ACTION cost with the shipped diversity floor disabled."
    },
    "actions_delta": {
        "principle": (
            "without_diversity - with_diversity; positive means diversity reached the "
            "first level-up in fewer actions."
        )
    },
    "solve_rate_with_diversity": {
        "principle": "held-out proxy solve-rate with CARNOT_ARC_EXPLORE_DIVERSITY=1."
    },
    "solve_rate_without_diversity": {
        "principle": "matched held-out proxy solve-rate with CARNOT_ARC_EXPLORE_DIVERSITY=0."
    },
    "diversity_off_control_passed": {
        "principle": (
            "the matched control arm ran and diversity-on first-win >= diversity-off."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "true with the diversity-off arm run as the matched control -- a "
            "no-transfer null is valid only then."
        )
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when firstwin_delta==0 -- states the equality is an honest "
            "no-transfer null, not a measurement bug."
        )
    },
    "offline_reproduced": {
        "principle": "any newly-reached win must offline-reproduce to count as a bank."
    },
    "chosen_submitted_config": {
        "principle": (
            "diversity is already shipped+wired; this confirms whether to keep it ON "
            "for the submitted agent (the A6 input)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _attempt_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and attempt.get("solved") is True


def _median(values: Sequence[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(float(value) for value in values))


def _median_actions(attempts: Sequence[Mapping[str, Any]]) -> float | None:
    return _median(exp4550.agent_actions_to_first_levelup(attempts))


def _first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_arcade") is not True:
        return "offline_arcade"
    if preconditions.get("arc_variant_generator_importable") is not True:
        return "arc_variant_generator_import"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission"
    return None


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    checks: JsonDict = {
        "offline_arcade": False,
        "arc_variant_generator_importable": False,
        "offline_env_public_games": exp4550._public_games(root_path),
        "leaderboard_submission": False,
        "required_commands": [
            '.venv/bin/python -c "from carnot.agentic import arc_solver_kit as k; '
            'k.offline_arcade()"',
            '.venv/bin/python -c "import carnot.agentic.arc_variant_generator"',
        ],
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_variant_generator import VariantEnv  # noqa: F401

        checks["arc_variant_generator_importable"] = True
    except Exception as exc:
        checks["arc_variant_generator_import_error"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = _first_precondition_miss(checks) is None
    return checks


@contextmanager
def _temporary_diversity(enabled: bool):
    old_value = os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY")
    os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("CARNOT_ARC_EXPLORE_DIVERSITY", None)
        else:
            os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = old_value


def make_variant_runner(mode: str, *, root: Path | str = REPO_ROOT) -> VariantRunner:
    """Run one manufactured variant with the shipped diversity flag forced on or off."""

    _root_path = Path(root)
    enabled = mode == "diversity_on"

    def run(game: str, spec: Mapping[str, Any], budget: int) -> JsonDict:
        with _temporary_diversity(enabled):
            attempt = dict(exp4550.default_variant_runner(game, spec, budget))
        attempt["diversity_mode"] = mode
        attempt["diversity_enabled"] = enabled
        attempt["diversity_env_var"] = "1" if enabled else "0"
        return attempt

    return run


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - live boundary
    return make_variant_runner(mode, root=REPO_ROOT)


def _measurement(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    runner: VariantRunner,
    n_bootstrap: int,
) -> JsonDict:
    measured = exp4550.measure_generic_transfer_over_variants(
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        variant_runner=runner,
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    measured["firstwin_count"] = int(measured["variant_solved_count"])
    measured["solve_rate"] = float(measured["generic_transfer_rate_over_variants"])
    measured["median_actions_to_first_levelup"] = _median_actions(measured["variant_attempts"])
    return measured


def _solved_by_signature(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(attempt.get("variant_signature")): attempt
        for attempt in attempts
        if attempt.get("attempted") is True
    }


def _gate_reproduced(gate: Any) -> bool:
    if not isinstance(gate, Mapping):
        return False
    claimed = max(1, _as_int(gate.get("claimed_level"), 1))
    reached = _as_int(gate.get("reached_level"), 0)
    return gate.get("reproduced") is True and reached >= claimed


def _newly_reached_reproduction(
    off_attempts: Sequence[Mapping[str, Any]],
    on_attempts: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str], list[JsonDict], bool]:
    off_by_sig = _solved_by_signature(off_attempts)
    newly_reached: list[str] = []
    unreproduced: list[str] = []
    records: list[JsonDict] = []
    for attempt in on_attempts:
        signature = str(attempt.get("variant_signature"))
        if not _attempt_solved(attempt) or _attempt_solved(off_by_sig.get(signature, {})):
            continue
        gate = attempt.get("reproduction_gate")
        reproduced = _gate_reproduced(gate)
        newly_reached.append(signature)
        if not reproduced:
            unreproduced.append(signature)
        gate_mapping = dict(gate) if isinstance(gate, Mapping) else {}
        records.append(
            {
                "game": str(attempt.get("game") or ""),
                "variant_signature": signature,
                "reached_level": _as_int(
                    gate_mapping.get("reached_level"), _as_int(attempt.get("reached_level"), 0)
                ),
                "solution_labels": list(attempt.get("solution_labels") or []),
                "reproduction_gate": gate_mapping,
                "reproduced": reproduced,
            }
        )
    return sorted(newly_reached), sorted(unreproduced), records, not unreproduced


def _registry_from_text(registry_text: str) -> JsonDict:
    loaded = yaml.safe_load(registry_text) if registry_text.strip() else {}
    return loaded if isinstance(loaded, dict) else {}


def _registry_game_levels(registry: Mapping[str, Any], game: str) -> int:
    for row in registry.get("games", []) or []:
        if isinstance(row, Mapping) and row.get("game") == game:
            return _as_int(row.get("levels_reproduced"), 0)
    return 0


def _registry_total(registry: Mapping[str, Any]) -> int:
    return _as_int(registry.get("reproducible_total_levels"), 0)


def _game_block_bounds(registry_text: str, game: str) -> tuple[int, int] | None:
    marker = f"- game: {game}"
    start = registry_text.find(marker)
    if start == -1:
        return None
    candidates = [
        index
        for index in (
            registry_text.find("\n- game: ", start + len(marker)),
            registry_text.find("\nreproducible_total_levels:", start + len(marker)),
        )
        if index != -1
    ]
    end = min(candidates) if candidates else len(registry_text)
    return start, end


def _replace_game_level(registry_text: str, game: str, reached_level: int) -> str:
    bounds = _game_block_bounds(registry_text, game)
    if bounds is None:
        block = (
            f"- game: {game}\n"
            "  reproducibility: reproduced\n"
            f"  levels_reproduced: {int(reached_level)}\n"
            f"  solver: {RESULT_RELATIVE_PATH}\n"
        )
        total_match = re.search(r"(?m)^reproducible_total_levels:\s*\d+\s*$", registry_text)
        if total_match:
            insert_at = total_match.start()
            prefix = registry_text[:insert_at]
            suffix = registry_text[insert_at:]
            separator = "" if prefix.endswith("\n") else "\n"
            return f"{prefix}{separator}{block}{suffix}"
        separator = "" if registry_text.endswith("\n") else "\n"
        return f"{registry_text}{separator}{block}"

    start, end = bounds
    block = registry_text[start:end]
    if re.search(r"(?m)^  levels_reproduced:\s*\d+\s*$", block):
        block = re.sub(
            r"(?m)^(  levels_reproduced:\s*)\d+\s*$",
            rf"\g<1>{int(reached_level)}",
            block,
            count=1,
        )
    else:
        first_newline = block.find("\n")
        if first_newline == -1:
            block = f"{block}\n  levels_reproduced: {int(reached_level)}"
        else:
            block = (
                f"{block[: first_newline + 1]}"
                f"  levels_reproduced: {int(reached_level)}\n"
                f"{block[first_newline + 1 :]}"
            )
    return f"{registry_text[:start]}{block}{registry_text[end:]}"


def _bankable_wins(
    registry: Mapping[str, Any], new_win_records: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    banks: list[JsonDict] = []
    for record in new_win_records:
        if record.get("reproduced") is not True and not _gate_reproduced(
            record.get("reproduction_gate")
        ):
            continue
        game = str(record.get("game") or "")
        reached = _as_int(record.get("reached_level"), 0)
        prior = _registry_game_levels(registry, game)
        if game and reached > prior:
            bank = dict(record)
            bank["prior_level"] = prior
            bank["banked_levels"] = reached - prior
            banks.append(bank)
    return banks


def apply_registry_banks(
    registry_text: str, new_win_records: Sequence[Mapping[str, Any]]
) -> tuple[str, JsonDict]:
    registry = _registry_from_text(registry_text)
    prior_total = _registry_total(registry)
    banks = _bankable_wins(registry, new_win_records)
    update: JsonDict = {
        "updated": False,
        "path": str(REGISTRY_RELATIVE_PATH),
        "banked_levels": 0,
        "prior_total_declared": prior_total,
        "new_total_declared": prior_total,
        "banked_wins": [],
        "reason": "no_new_reproduced_level",
    }
    if not banks:
        return registry_text, update

    updated_text = registry_text
    total_banked = 0
    for bank in banks:
        total_banked += int(bank["banked_levels"])
        updated_text = _replace_game_level(
            updated_text, str(bank["game"]), int(bank["reached_level"])
        )
    new_total = prior_total + total_banked
    if re.search(r"(?m)^reproducible_total_levels:\s*\d+\s*$", updated_text):
        updated_text = re.sub(
            r"(?m)^(reproducible_total_levels:\s*)\d+\s*$",
            rf"\g<1>{new_total}",
            updated_text,
            count=1,
        )
    else:
        separator = "" if updated_text.endswith("\n") else "\n"
        updated_text = f"{updated_text}{separator}reproducible_total_levels: {new_total}\n"

    update.update(
        {
            "updated": True,
            "banked_levels": total_banked,
            "new_total_declared": new_total,
            "banked_wins": banks,
            "reason": "banked_offline_reproduced_diversity_win",
        }
    )
    return updated_text, update


def _registry_update(
    root: Path,
    new_win_records: Sequence[Mapping[str, Any]],
    *,
    update_registry: bool,
) -> JsonDict:
    if not new_win_records:
        return {
            "updated": False,
            "path": str(REGISTRY_RELATIVE_PATH),
            "banked_levels": 0,
            "banked_wins": [],
            "reason": "no_new_diversity_wins",
        }
    if not update_registry:
        return {
            "updated": False,
            "path": str(REGISTRY_RELATIVE_PATH),
            "banked_levels": 0,
            "banked_wins": [],
            "reason": "registry_update_disabled",
        }
    registry_path = root / REGISTRY_RELATIVE_PATH
    if not registry_path.exists():
        return {
            "updated": False,
            "path": str(REGISTRY_RELATIVE_PATH),
            "banked_levels": 0,
            "banked_wins": [],
            "reason": "registry_missing",
        }
    registry_text = registry_path.read_text(encoding="utf-8")
    updated_text, update = apply_registry_banks(registry_text, new_win_records)
    if update.get("updated") is True:
        registry_path.write_text(updated_text, encoding="utf-8")
    return update


def _checksum(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "firstwin_count_diversity_on": artifact.get("firstwin_count_diversity_on"),
        "firstwin_count_diversity_off": artifact.get("firstwin_count_diversity_off"),
        "firstwin_delta": artifact.get("firstwin_delta"),
        "median_actions_to_first_levelup_with_diversity": artifact.get(
            "median_actions_to_first_levelup_with_diversity"
        ),
        "actions_delta": artifact.get("actions_delta"),
        "solve_rate_with_diversity": artifact.get("solve_rate_with_diversity"),
        "solve_rate_without_diversity": artifact.get("solve_rate_without_diversity"),
        "diversity_off_control_passed": artifact.get("diversity_off_control_passed"),
        "offline_reproduced": artifact.get("offline_reproduced"),
        "newly_reached_wins": artifact.get("newly_reached_wins"),
        "registry_update": artifact.get("registry_update"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "variant_plan": artifact.get("variant_plan"),
    }


def _blocked_artifact(
    *,
    resource: str,
    preconditions: Mapping[str, Any],
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4583",
            "SCENARIO-CAPSTONE-4583",
            "SCENARIO-CAPSTONE-4583-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"complete: blocked_{resource}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "firstwin_count_diversity_on": 0,
        "firstwin_count_diversity_off": 0,
        "firstwin_delta": 0,
        "median_actions_to_first_levelup_with_diversity": None,
        "median_actions_to_first_levelup_without_diversity": None,
        "actions_delta": 0.0,
        "solve_rate_with_diversity": 0.0,
        "solve_rate_without_diversity": 0.0,
        "diversity_off_control_passed": False,
        "false_negative_risk_checked": False,
        "null_delta_methodology_note": (
            "blocked before paired diversity measurement; no firstwin delta was fabricated."
        ),
        "offline_reproduced": False,
        "chosen_submitted_config": "leave_diversity_floor_default_off",
        "newly_reached_wins": [],
        "unreproduced_new_wins": [],
        "new_win_reproduction_records": [],
        "new_levels_banked": 0,
        "registry_update": {
            "updated": False,
            "path": str(REGISTRY_RELATIVE_PATH),
            "banked_levels": 0,
            "banked_wins": [],
            "reason": f"blocked_{resource}",
        },
        "preconditions_checked": dict(preconditions),
        "variant_plan": {
            "public_games": sorted(str(game) for game in public_games),
            "public_game_count": len(public_games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
            "arms": ["diversity_off", "diversity_on"],
        },
        "diversity_off_measurement": {},
        "diversity_on_measurement": {},
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner_factory: VariantRunnerFactory = default_variant_runner_factory,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    update_registry: bool = False,
) -> JsonDict:
    started = time.monotonic()
    root_path = Path(root)
    preconditions = dict(preconditions_checked or check_preconditions(root_path))
    games = list(public_games or preconditions.get("offline_env_public_games") or [])
    miss = _first_precondition_miss(preconditions)
    if miss:
        return _blocked_artifact(
            resource=miss,
            preconditions=preconditions,
            public_games=games,
            variant_ids=variant_ids,
            budget=budget,
        )

    diversity_off = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("diversity_off"),
        n_bootstrap=n_bootstrap,
    )
    diversity_on = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("diversity_on"),
        n_bootstrap=n_bootstrap,
    )

    off_count = int(diversity_off["firstwin_count"])
    on_count = int(diversity_on["firstwin_count"])
    firstwin_delta = on_count - off_count
    off_actions = diversity_off["median_actions_to_first_levelup"]
    on_actions = diversity_on["median_actions_to_first_levelup"]
    actions_delta = (
        round(float(off_actions) - float(on_actions), 10)
        if off_actions is not None and on_actions is not None
        else 0.0
    )
    off_completed = int(diversity_off["variant_attempts_count"]) > 0
    diversity_off_control_passed = bool(off_completed and on_count >= off_count)
    false_negative_risk_checked = bool(diversity_off_control_passed)
    newly_reached, unreproduced, reproduction_records, offline_reproduced = (
        _newly_reached_reproduction(
            diversity_off["variant_attempts"], diversity_on["variant_attempts"]
        )
    )
    registry_update = _registry_update(
        root_path, reproduction_records, update_registry=update_registry
    )
    new_levels_banked = int(registry_update.get("banked_levels") or 0)

    if firstwin_delta == 0 and diversity_off_control_passed:
        null_note = (
            "firstwin_delta==0 is an honest no-transfer null under the paired "
            "same-variant diversity-off control, not a measurement bug."
        )
    elif firstwin_delta == 0:
        null_note = (
            "firstwin_delta==0 but the matched diversity-off control did not pass, "
            "so false-negative risk remains open."
        )
    else:
        null_note = ""

    if firstwin_delta > 0 and diversity_off_control_passed and offline_reproduced:
        verdict = f"success: diversity_floor_transfer_firstwin_up_{firstwin_delta}"
    elif firstwin_delta > 0 and not offline_reproduced:
        verdict = "complete: diversity_floor_new_win_unreproduced_no_bank"
    elif firstwin_delta == 0 and diversity_off_control_passed:
        verdict = "complete: diversity_floor_no_transfer_honest_null_gap_sharpened"
    elif not diversity_off_control_passed:
        verdict = "complete: diversity_floor_regression_control_failed_false_negative_risk_open"
    else:  # pragma: no cover - defensive fallback for future verdict states.
        verdict = "complete: diversity_floor_no_transfer_control_incomplete"

    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4583",
            "SCENARIO-CAPSTONE-4583",
            "SCENARIO-CAPSTONE-4583-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "firstwin_count_diversity_on": on_count,
        "firstwin_count_diversity_off": off_count,
        "firstwin_delta": int(firstwin_delta),
        "median_actions_to_first_levelup_with_diversity": on_actions,
        "median_actions_to_first_levelup_without_diversity": off_actions,
        "actions_delta": float(actions_delta),
        "solve_rate_with_diversity": float(diversity_on["solve_rate"]),
        "solve_rate_without_diversity": float(diversity_off["solve_rate"]),
        "diversity_off_control_passed": diversity_off_control_passed,
        "false_negative_risk_checked": false_negative_risk_checked,
        "null_delta_methodology_note": null_note,
        "offline_reproduced": bool(offline_reproduced),
        "chosen_submitted_config": (
            "keep_diversity_floor_on"
            if verdict.startswith("success:")
            else "leave_diversity_floor_default_off"
        ),
        "newly_reached_wins": newly_reached,
        "unreproduced_new_wins": unreproduced,
        "new_win_reproduction_records": reproduction_records,
        "new_levels_banked": new_levels_banked,
        "registry_update": registry_update,
        "preconditions_checked": preconditions,
        "variant_plan": {
            "public_games": sorted(str(game) for game in games),
            "public_game_count": len(games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
            "runner": "paired_diversity_on_off_generic_solver_offline_variant_env",
            "arms": ["diversity_off", "diversity_on"],
            "value_head_best_first_expansion": False,
        },
        "diversity_off_measurement": diversity_off,
        "diversity_on_measurement": diversity_on,
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.monotonic() - started, 6),
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def _type_name(field: str) -> str:
    return f"{field} must be a bare int"


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in (
        "firstwin_count_diversity_on",
        "firstwin_count_diversity_off",
        "firstwin_delta",
        "random_seed",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(_type_name(field))
    for field in (
        "median_actions_to_first_levelup_with_diversity",
        "median_actions_to_first_levelup_without_diversity",
    ):
        value = artifact.get(field)
        if value is not None and type(value) is not float:
            errors.append(f"{field} must be float or null")
    for field in (
        "actions_delta",
        "solve_rate_with_diversity",
        "solve_rate_without_diversity",
    ):
        if type(artifact.get(field)) is not float:
            errors.append(f"{field} must be a bare float")
    for field in (
        "diversity_off_control_passed",
        "false_negative_risk_checked",
        "offline_reproduced",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be a bare bool")
    if artifact.get("firstwin_delta") == 0 and not artifact.get(
        "null_delta_methodology_note"
    ):
        errors.append("null_delta_methodology_note required for zero firstwin_delta")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles missing")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"missing field principle for {field}")
    return errors


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> Path:
    root_path = Path(root)
    payload = dict(artifact or build_artifact(root_path))
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    path = root_path / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path | str = REPO_ROOT,
    *,
    write: bool = True,
    update_registry: bool = True,
) -> JsonDict:
    artifact = build_artifact(root, update_registry=update_registry)
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True, update_registry=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
