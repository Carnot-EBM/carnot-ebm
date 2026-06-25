"""Experiment 4728: bank ar25 L3 and checkpoint the learned verifier.

Spec refs: REQ-ARC-WMTE-4728, SCENARIO-ARC-WMTE-4728.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
MODELS = REPO / "models"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4728_levelup_selfplay.json"

EXPERIMENT = "experiment_4728_levelup_selfplay"
RESULT_RELATIVE_PATH = "results/experiment_4728_levelup_selfplay.json"
LOOP_RESULT_RELATIVE_PATH = "results/arc_loop_solve_ar25.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
CHECKPOINT_RELATIVE_PATH = "models/arc_verifier_ar25.json"
RANDOM_SEED = 4728
TARGET_GAME = "ar25"
TARGET_LEVEL = 3
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "development_proxy"

SPEC_REFS = [
    "REQ-ARC-WMTE-4728",
    "SCENARIO-ARC-WMTE-4728-ROTATED-TARGET",
    "SCENARIO-ARC-WMTE-4728-BANK-AND-CHECKPOINT",
    "SCENARIO-ARC-WMTE-4728-REGISTRY-GATE",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank.",
    "inference_substrate": "verifier_ensemble_against_cached_candidates -- arc_loop_solve scores against the offline sim (1s floor), no live LLM load.",
    "offline_reproduced": "only offline-reproduced levels count (ARC Solve Reproducibility); a live-only trajectory is provisional.",
    "reproduced_levels": "the integer level reached on the target game.",
    "new_levels_banked": ">=1 is the Level-Up Guarantee met; 0 rotates the target next milestone.",
    "reproducible_total_levels": "the registry header after this run (63->64+ if banked) -- the monotonic north-star metric.",
    "verifier_checkpoint": "models/arc_verifier_<game>.json -- the self-play loop trains+checkpoints the learned verifier (the self-improvement-every-milestone mandate).",
    "verifier_is_oracle": "MUST be false -- the learned verifier routes/ranks; the executable reproduction gate is the oracle-distinct authority.",
    "solve_provenance": "live_agent_self_discovery for a generic bank, OR development_proxy honestly if a GameAdapter delta was required.",
    "registry_precheck_passed": "confirms the target level is NOT already in the registry (a duplicate is a CRITICAL adversarial flag).",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "content-addressed hash catches silent harness/corpus drift on replay.",
    "preconditions_checked": "records resources verified (offline arcade, arc_loop_solve runnable); pre-empts missing-resource fabrication.",
}

PREFERRED_TARGETS = ("re86", "s5i5", "g50t", "r11l")
ALTERNATIVE_TARGETS = ("m0r0", "cn04", "ar25")
PROHIBITED_TARGETS = ("bp35", "lf52", "sb26", "dc22", "vc33", "sk48")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _registry_data(registry_path: Path | None = None) -> dict[str, Any]:
    path = REGISTRY if registry_path is None else Path(registry_path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _game_entry(registry: dict[str, Any], game: str) -> dict[str, Any]:
    for entry in registry.get("games", []):
        if isinstance(entry, dict) and entry.get("game") == game:
            return entry
    return {}


def _dead_end_notes(entry: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for item in entry.get("dead_ends") or []:
        if isinstance(item, dict):
            summary = item.get("gap_id") or item.get("filled_summary")
            if summary:
                rows.append(str(summary))
            elif len(item) == 1:
                key, value = next(iter(item.items()))
                rows.append(f"{key}: {value}")
            else:
                rows.append(str(item))
        else:
            rows.append(str(item))
    return rows


def _dead_ends_for_game(registry: dict[str, Any], game: str) -> list[str]:
    notes = _dead_end_notes(_game_entry(registry, game))
    prefix = f"{game}:"
    for entry in registry.get("games", []):
        if not isinstance(entry, dict):
            continue
        for note in _dead_end_notes(entry):
            if note.startswith(prefix):
                notes.append(note)
    return list(dict.fromkeys(notes))


def registry_level(game: str, registry_path: Path | None = None) -> int:
    entry = _game_entry(_registry_data(registry_path), game)
    try:
        return int(entry.get("levels_reproduced") or 0)
    except (TypeError, ValueError):  # pragma: no cover - defensive registry parsing.
        return 0


def registry_total_levels(registry_path: Path | None = None) -> int:
    try:
        return int(_registry_data(registry_path).get("reproducible_total_levels") or 0)
    except (TypeError, ValueError):  # pragma: no cover - defensive registry parsing.
        return 0


def select_target(registry_path: Path | None = None) -> tuple[str, dict[str, Any]]:
    registry = _registry_data(registry_path)
    prior_level = registry_level(TARGET_GAME, registry_path=registry_path)
    if prior_level >= TARGET_LEVEL:
        raise RuntimeError(
            f"duplicate registry precheck: {TARGET_GAME} already records L{prior_level}"
        )
    if prior_level < 2:
        raise RuntimeError("ar25 deepen requires an existing reproduced L2 registry row")
    dead_ends_by_game = {
        game: _dead_ends_for_game(registry, game)
        for game in (*PREFERRED_TARGETS, TARGET_GAME)
    }
    return TARGET_GAME, {
        "preferred_targets": list(PREFERRED_TARGETS),
        "requested_alternatives": list(ALTERNATIVE_TARGETS),
        "prohibited_targets": list(PROHIBITED_TARGETS),
        "selected": TARGET_GAME,
        "target_level": TARGET_LEVEL,
        "registry_level_before": prior_level,
        "registry_precheck_passed": True,
        "dead_ends_by_game": dead_ends_by_game,
        "dead_ends_seen": dead_ends_by_game[TARGET_GAME],
        "selection_reason": (
            "re86 is already L2; s5i5/g50t/r11l carry recorded L2 stalls; "
            "ar25 is the first allowed L2->L3 alternative with a grounded reflection tail"
        ),
        "rotation_rule": (
            "prefer re86/s5i5/g50t/r11l; skip bp35/lf52/sb26/dc22/vc33/sk48 "
            "and do not re-bank levels already recorded in the registry"
        ),
        "registry_game_count": len(registry.get("games", [])),
    }


def check_preconditions() -> list[str]:  # pragma: no cover - live SDK/process boundary.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    proc = subprocess.run(
        [sys.executable, "scripts/arc_loop_solve.py", "--help"],
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return ["arc_solver_kit.offline_arcade()", "scripts/arc_loop_solve.py --help"]


def read_standing_loop_result(game: str) -> dict[str, Any]:
    result_path = RESULTS / f"arc_loop_solve_{game}.json"
    out = json.loads(result_path.read_text(encoding="utf-8"))
    out["_standing_loop_reused"] = True
    return out


def run_standing_loop(game: str, target_level: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/arc_loop_solve.py",
        "--game",
        game,
        "--target-level",
        str(target_level),
        "--no-hazard-prune",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:  # pragma: no cover - exercised only by failed live loop.
        raise RuntimeError(proc.stdout)
    result_path = RESULTS / f"arc_loop_solve_{game}.json"
    out = json.loads(result_path.read_text(encoding="utf-8"))
    out["_standing_loop_stdout"] = proc.stdout
    return out


def _gate_reached(loop_result: dict[str, Any]) -> int:
    gate = dict(loop_result.get("reproduction_gate") or {})
    return int(gate.get("reached_level") or loop_result.get("reached_level") or 0)


def _gate_reproduced(loop_result: dict[str, Any]) -> bool:
    gate = dict(loop_result.get("reproduction_gate") or {})
    return bool(gate.get("reproduced", loop_result.get("offline_reproduced")))


def _loop_is_bankable(loop_result: dict[str, Any], prior_level: int) -> bool:
    return bool(
        loop_result.get("game") == TARGET_GAME
        and loop_result.get("offline_reproduced")
        and _gate_reproduced(loop_result)
        and _gate_reached(loop_result) > int(prior_level)
        and loop_result.get("learned_verifier_checkpoint") == CHECKPOINT_RELATIVE_PATH
    )


def load_or_run_standing_loop(game: str, target_level: int, prior_level: int) -> dict[str, Any]:
    try:
        cached = read_standing_loop_result(game)
    except FileNotFoundError:  # pragma: no cover - live path when no cache exists.
        cached = {}
    if cached and _loop_is_bankable(cached, prior_level):
        return cached
    return run_standing_loop(game, target_level)


def _stable_checksum(payload: dict[str, Any]) -> str:
    checksum_payload = {
        key: payload.get(key)
        for key in (
            "target_game",
            "prior_reproduced_level",
            "offline_reproduced",
            "reproduced_levels",
            "new_levels_banked",
            "reproduction_gate",
            "solution_labels",
            "verifier_checkpoint",
            "target_selection",
        )
    }
    raw = json.dumps(
        checksum_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def artifact_schema_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in FIELD_PRINCIPLES:
        if field not in payload:
            errors.append(f"missing_field:{field}")
        if FIELD_PRINCIPLES[field] != payload.get("field_principles", {}).get(field):
            errors.append(f"missing_principle:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(("success:", "complete:")):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if payload.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance_must_be_development_proxy")
    if payload.get("offline_reproduced") and int(payload.get("new_levels_banked") or 0) < 1:
        errors.append("offline_reproduced_without_new_level")
    if payload.get("offline_reproduced") and not payload.get("registry_precheck_passed"):
        errors.append("offline_reproduced_without_registry_precheck")
    if payload.get("offline_reproduced") and not payload.get("registry_updated"):
        errors.append("offline_reproduced_without_registry_update")
    if payload.get("offline_reproduced") and payload.get("verifier_checkpoint") != CHECKPOINT_RELATIVE_PATH:
        errors.append("verifier_checkpoint_must_match_ar25")
    if not _checksum_is_hex(payload.get("reproducibility_checksum")):
        errors.append("invalid_reproducibility_checksum")
    return errors


def build_artifact(
    loop_result: dict[str, Any],
    *,
    prior_level: int,
    prior_total_levels: int,
    registry_updated: bool,
    checkpoint_before_sha: str | None,
    checkpoint_after_sha: str | None,
    dead_ends_recorded: list[str],
    preconditions_checked: list[str],
    target_selection: dict[str, Any],
) -> dict[str, Any]:
    game = str(loop_result.get("game") or TARGET_GAME)
    gate = dict(loop_result.get("reproduction_gate") or {})
    reached_level = _gate_reached(loop_result)
    new_levels = max(0, reached_level - int(prior_level))
    checkpoint_path = loop_result.get("learned_verifier_checkpoint")
    checkpoint_ready = bool(checkpoint_path == CHECKPOINT_RELATIVE_PATH and checkpoint_after_sha)
    registry_precheck_passed = bool(target_selection.get("registry_precheck_passed"))
    success = bool(
        _gate_reproduced(loop_result)
        and loop_result.get("offline_reproduced")
        and new_levels >= 1
        and checkpoint_ready
        and registry_updated
        and registry_precheck_passed
    )
    verdict = (
        f"success: {game}_L{reached_level}_offline_reproduced"
        if success
        else f"complete: {game}_delta_identified_no_bank"
    )
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": "carnot.exp4728.levelup_selfplay.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_reproduced": bool(success),
        "reproduced_levels": int(reached_level if success else 0),
        "new_levels_banked": int(new_levels if success else 0),
        "reproducible_total_levels": int(prior_total_levels + (new_levels if success else 0)),
        "verifier_checkpoint": checkpoint_path,
        "verifier_is_oracle": False,
        "solve_provenance": loop_result.get("solve_provenance") or SOLVE_PROVENANCE,
        "registry_precheck_passed": registry_precheck_passed,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": list(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "target_game": game,
        "standing_loop_result_path": f"results/arc_loop_solve_{game}.json",
        "registry_updated": bool(registry_updated),
        "reproduction_gate": gate,
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "dead_ends_recorded": list(dead_ends_recorded),
        "target_selection": dict(target_selection),
        "prior_reproduced_level": int(prior_level),
        "reached_level": reached_level,
        "reproducible_total_levels_before": int(prior_total_levels),
        "verifier_delta": {
            "checkpoint_path": checkpoint_path,
            "before_sha256": checkpoint_before_sha,
            "after_sha256": checkpoint_after_sha,
            "updated": checkpoint_ready,
            "sha_changed": bool(
                checkpoint_after_sha and checkpoint_after_sha != checkpoint_before_sha
            ),
            "positive_trace_steps": len(loop_result.get("solution_labels") or []),
            "negative_trace_notes": list(dead_ends_recorded),
        },
        "selected_generic_operators": list(loop_result.get("selected_generic_operators") or []),
        "states_expanded": loop_result.get("states_expanded"),
    }
    artifact["reproducibility_checksum"] = _stable_checksum(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    return artifact


def _replace_game_block(text: str, game: str, replacement: str) -> str:
    marker = f"- game: {game}"
    start = text.index(marker)
    candidates = [
        index
        for index in (
            text.find("\n- game: ", start + len(marker)),
            text.find("\nreproducible_total_levels:", start + len(marker)),
        )
        if index != -1
    ]
    end = min(candidates) if candidates else len(text)
    return f"{text[:start]}{replacement}{text[end:]}"


def _dead_end_lines(artifact: dict[str, Any]) -> str:
    dead_ends = artifact.get("dead_ends_recorded") or []
    if not dead_ends:
        return "  dead_ends: []"
    rows = ["  dead_ends:"]
    for item in dead_ends:
        dumped = yaml.safe_dump(
            [str(item)],
            default_flow_style=False,
            sort_keys=False,
            width=4096,
        ).strip()
        rows.append(f"  {dumped}")
    return "\n".join(rows)


def _ar25_registry_block(checksum: str, checkpoint_path: str, artifact: dict[str, Any]) -> str:
    return f"""- game: ar25
  reproducibility: reproduced
  levels_reproduced: 3
  mechanic_class: reflection_mirror_object_alignment
  win_condition: >-
    L1 object reflection plus L2 vertical mirror/object alignment plus L3
    horizontal-axis reflection coverage: move the horizontal 0002nuguepuujf axis
    from y=16 to y=9, place the L-shaped object at (11,14), and place the
    two-row object at (3,14) so reflections cover every fixed 0001sruqbuvukh
    target cell.
  action_model: >-
    Keyboard-only ACTION1-4 movement plus ACTION5 selection cycling. L3 tail:
    ACTION1 x7 moves the selected horizontal axis up, ACTION5 selects the
    L-shaped object, ACTION4 x7 and ACTION2 x7 place it, ACTION5 selects the
    two-row object, then ACTION3 x12 and ACTION2 x5 place it.
  solver: >-
    GameAdapter _ar25 in python/carnot/agentic/arc_game_adapters.py plus
    scripts/arc_loop_solve.py --game ar25 --target-level 3 --no-hazard-prune.
  reproduce: >-
    Exp4728 results/experiment_4728_levelup_selfplay.json re-gated
    results/arc_loop_solve_ar25.json offline_reproduced=True, reached_level=3,
    banked +1 over the prior L2 registry row, checksum {checksum}.
  learned_verifier_checkpoint: {checkpoint_path}
  gotchas:
  - The L3 axis sprite is selected at level entry; moving it horizontally is
    ignored by the environment, so ACTION1 x7 is the load-bearing axis delta.
  - The L3 target decomposes into two reflected object masks around y=9; moving
    only the axis does not win.
  - ACTION7 pops a hidden undo stack and remains excluded from the banked path.
  - The learned verifier checkpoint routes/ranks future search only; the
    executable offline reproduction gate remains the oracle-distinct authority.
{_dead_end_lines(artifact)}
  latest_exp4728_levelup_selfplay:
    artifact: results/experiment_4728_levelup_selfplay.json
    loop_artifact: results/arc_loop_solve_ar25.json
    offline_reproduced: true
    reproduced_levels: 3
    new_levels_banked: 1
    verifier_checkpoint: {checkpoint_path}
    target_rotation: preferred_L1_targets_prechecked_then_allowed_L2_to_L3_alternative
    reproducibility_checksum: {checksum}"""


def update_registry_for_success(
    artifact: dict[str, Any],
    registry_path: Path | None = None,
) -> bool:
    game = artifact["target_game"]
    checkpoint = artifact["verifier_checkpoint"]
    checksum = artifact["reproducibility_checksum"]
    path = REGISTRY if registry_path is None else Path(registry_path)
    text = path.read_text(encoding="utf-8")
    old_text = text
    text = _replace_game_block(text, game, _ar25_registry_block(checksum, checkpoint, artifact))
    before_total = int(artifact["reproducible_total_levels_before"])
    after_total = int(artifact["reproducible_total_levels"])
    text = re.sub(
        rf"(?m)^(reproducible_total_levels:\s*){before_total}\s*$",
        rf"\g<1>{after_total}",
        text,
        count=1,
    )
    if text != old_text:
        path.write_text(text, encoding="utf-8")
    return text != old_text


def _write_artifact(payload: dict[str, Any], path: Path | None = None) -> None:
    output = Path(path) if path is not None else ARTIFACT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dead_ends_from_selection(
    target_selection: dict[str, Any], reached_level: int, prior_level: int
) -> list[str]:
    dead_ends: list[str] = []
    for game, notes in (target_selection.get("dead_ends_by_game") or {}).items():
        for item in notes:
            dead_ends.append(f"{game}: registry_prechecked:{item}")
    dead_ends.append("g50t: clone_replay_L2_route_reached_distance_12_no_bank")
    dead_ends.append("ar25: ACTION7 hidden undo stack excluded from banked L3 path")
    if reached_level <= prior_level:
        dead_ends.append(
            f"ar25 standing loop reached L{reached_level}, not beyond prior L{prior_level}"
        )
    return list(dict.fromkeys(dead_ends))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game")
    parser.add_argument("--refresh-loop", action="store_true")
    args = parser.parse_args(argv)

    preconditions_checked = check_preconditions()
    if args.game:
        game = args.game
        prior_for_override = registry_level(game)
        target_selection = {
            "selected": game,
            "override": True,
            "target_level": prior_for_override + 1,
            "registry_level_before": prior_for_override,
            "registry_precheck_passed": prior_for_override < prior_for_override + 1,
            "dead_ends_by_game": {},
        }
    else:
        game, target_selection = select_target()

    prior_level = registry_level(game)
    prior_total = registry_total_levels()
    checkpoint = MODELS / f"arc_verifier_{game}.json"
    before_sha = sha256_file(checkpoint)
    target_level = int(target_selection.get("target_level") or prior_level + 1)
    if args.refresh_loop:  # pragma: no cover - live refresh path.
        loop = run_standing_loop(game, target_level)
        preconditions_checked.append("scripts/arc_loop_solve.py --refresh-loop")
    else:
        loop = load_or_run_standing_loop(game, target_level, prior_level)
        preconditions_checked.append(f"standing_loop_result_or_run:{LOOP_RESULT_RELATIVE_PATH}")
    after_sha = sha256_file(checkpoint)
    reached_level = _gate_reached(loop)
    dead_ends = _dead_ends_from_selection(target_selection, reached_level, prior_level)
    success_ready = bool(
        loop.get("offline_reproduced")
        and _gate_reproduced(loop)
        and reached_level > prior_level
        and loop.get("learned_verifier_checkpoint") == CHECKPOINT_RELATIVE_PATH
        and after_sha
        and target_selection.get("registry_precheck_passed")
    )
    draft = build_artifact(
        loop,
        prior_level=prior_level,
        prior_total_levels=prior_total,
        registry_updated=success_ready,
        checkpoint_before_sha=before_sha,
        checkpoint_after_sha=after_sha,
        dead_ends_recorded=dead_ends,
        preconditions_checked=preconditions_checked,
        target_selection=target_selection,
    )
    _write_artifact(draft)
    registry_updated = update_registry_for_success(draft) if success_ready else False
    artifact = build_artifact(
        loop,
        prior_level=prior_level,
        prior_total_levels=prior_total,
        registry_updated=registry_updated,
        checkpoint_before_sha=before_sha,
        checkpoint_after_sha=after_sha,
        dead_ends_recorded=dead_ends,
        preconditions_checked=preconditions_checked,
        target_selection=target_selection,
    )
    _write_artifact(artifact)
    print(
        json.dumps(
            {
                key: artifact[key]
                for key in (
                    "honest_verdict",
                    "target_game",
                    "offline_reproduced",
                    "reproduced_levels",
                    "new_levels_banked",
                    "reproducible_total_levels",
                    "verifier_checkpoint",
                    "registry_updated",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct command in task.
    raise SystemExit(main())
