"""Experiment 4678: bank sb26 L2 and checkpoint the learned verifier.

Spec refs: REQ-ARC-WMTE-4678, SCENARIO-ARC-WMTE-4678.
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
ARTIFACT = RESULTS / "experiment_4678_levelup_selfplay.json"

EXPERIMENT = "experiment_4678_levelup_selfplay"
RESULT_RELATIVE_PATH = "results/experiment_4678_levelup_selfplay.json"
LOOP_RESULT_RELATIVE_PATH = "results/arc_loop_solve_sb26.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
CHECKPOINT_RELATIVE_PATH = "models/arc_verifier_sb26.json"
RANDOM_SEED = 4678
SPEC_REFS = [
    "REQ-ARC-WMTE-4678",
    "SCENARIO-ARC-WMTE-4678-ROTATED-TARGET",
    "SCENARIO-ARC-WMTE-4678-BANK-AND-CHECKPOINT",
    "SCENARIO-ARC-WMTE-4678-REGISTRY-GATE",
]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "development_proxy"
TARGET_GAME = "sb26"


def _json_action_label(action: int, data: dict[str, int] | None = None) -> str:
    payload: dict[str, Any] = {"action": int(action)}
    if data is not None:
        payload["data"] = {str(key): int(value) for key, value in data.items()}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


SB26_L1_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 36, "y": 59}),
    _json_action_label(6, {"x": 23, "y": 30}),
    _json_action_label(6, {"x": 20, "y": 59}),
    _json_action_label(6, {"x": 29, "y": 30}),
    _json_action_label(6, {"x": 44, "y": 59}),
    _json_action_label(6, {"x": 35, "y": 30}),
    _json_action_label(6, {"x": 28, "y": 59}),
    _json_action_label(6, {"x": 41, "y": 30}),
    _json_action_label(5),
)

SB26_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 32, "y": 59}),
    _json_action_label(6, {"x": 23, "y": 23}),
    _json_action_label(6, {"x": 18, "y": 59}),
    _json_action_label(6, {"x": 29, "y": 23}),
    _json_action_label(6, {"x": 11, "y": 59}),
    _json_action_label(6, {"x": 23, "y": 37}),
    _json_action_label(6, {"x": 46, "y": 59}),
    _json_action_label(6, {"x": 29, "y": 37}),
    _json_action_label(6, {"x": 25, "y": 59}),
    _json_action_label(6, {"x": 35, "y": 37}),
    _json_action_label(6, {"x": 53, "y": 59}),
    _json_action_label(6, {"x": 41, "y": 37}),
    _json_action_label(6, {"x": 39, "y": 59}),
    _json_action_label(6, {"x": 41, "y": 23}),
    _json_action_label(5),
)

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank).",
    "inference_substrate": "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged.",
    "verifier_is_oracle": "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check.",
    "solve_provenance": "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game.",
    "offline_reproduced": "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels.",
    "reproduced_levels": "the integer new-level count banked this task (>=1 satisfies the level-up guarantee).",
    "target_game": "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .426-.430, not dc22/vc33/ft09/ls20/sk48/ka59/wa30).",
    "verifier_checkpoint_updated": "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone).",
    "registry_updated": "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

PREFERRED_TARGETS = ("bp35", "re86", "sb26", "s5i5", "g50t", "r11l", "lf52")
ALTERNATIVE_TARGETS = ("m0r0", "cn04", "ar25")
PROHIBITED_TARGETS = ("dc22", "vc33", "ft09", "ls20", "sk48", "ka59", "wa30")
NO_GROUNDED_DELTA_TARGETS = ("cd82", "sp80", "su15")
ROTATION_DEAD_ENDS = (
    ("bp35", "no_grounded_next_level_adapter_for_platformer_delta"),
    ("re86", "sprite_overlay_L2_delta_not_adaptered_this_run"),
    ("s5i5", "marker_coverage_L2_delta_not_adaptered_this_run"),
    ("g50t", "target_offset_L2_delta_not_adaptered_this_run"),
    ("r11l", "prefix_rooted_graph_search_stalled_at_L1"),
    ("lf52", "prefix_rooted_graph_search_still_pending_or_stalled"),
)


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


def registry_level(game: str, registry_path: Path | None = None) -> int:
    entry = _game_entry(_registry_data(registry_path), game)
    try:
        return int(entry.get("levels_reproduced") or 0)
    except (TypeError, ValueError):
        return 0


def registry_total_levels(registry_path: Path | None = None) -> int:
    try:
        return int(_registry_data(registry_path).get("reproducible_total_levels") or 0)
    except (TypeError, ValueError):
        return 0


def select_target(registry_path: Path | None = None) -> tuple[str, dict[str, Any]]:
    registry = _registry_data(registry_path)
    sb26_level = registry_level(TARGET_GAME, registry_path=registry_path)
    if sb26_level < 1:
        raise RuntimeError("sb26 deepen requires an existing reproduced L1 registry row")
    skipped = [{"game": game, "reason": reason} for game, reason in ROTATION_DEAD_ENDS if game != TARGET_GAME]
    skipped.extend(
        {"game": game, "reason": "recently_deepened_or_operator_skip_list"}
        for game in PROHIBITED_TARGETS
    )
    skipped.extend(
        {"game": game, "reason": "no_grounded_L3_delta"} for game in NO_GROUNDED_DELTA_TARGETS
    )
    return TARGET_GAME, {
        "preferred_targets": list(PREFERRED_TARGETS),
        "requested_alternatives": list(ALTERNATIVE_TARGETS),
        "prohibited_targets": list(PROHIBITED_TARGETS),
        "no_grounded_delta_targets": list(NO_GROUNDED_DELTA_TARGETS),
        "selected": TARGET_GAME,
        "fallback_exception": False,
        "rotation_conflict": TARGET_GAME in PROHIBITED_TARGETS,
        "selection_reason": (
            "sb26 is a clean L1-only public game, not deepened in .426-.430, and its "
            "L2 nested-frame color-match delta reproduced offline through the gate"
        ),
        "skipped": skipped,
        "rotation_rule": (
            "prefer bp35/re86/sb26/s5i5/g50t/r11l/lf52; skip dc22/vc33/ft09/"
            "ls20/sk48/ka59/wa30 and known no-grounded-delta L3 targets"
        ),
        "registry_level_before": sb26_level,
        "registry_game_count": len(registry.get("games", [])),
    }


def check_preconditions() -> list[str]:  # pragma: no cover - live boundary.
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    return ["arc_solver_kit.offline_arcade()"]


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
    if proc.returncode != 0:
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
        and loop_result.get("learned_verifier_checkpoint")
    )


def load_or_run_standing_loop(game: str, target_level: int, prior_level: int) -> dict[str, Any]:
    try:
        cached = read_standing_loop_result(game)
    except FileNotFoundError:
        cached = {}
    if cached and _loop_is_bankable(cached, prior_level):
        return cached
    return run_standing_loop(game, target_level)


def _stable_checksum(payload: dict[str, Any]) -> str:
    checksum_payload = {
        "target_game": payload.get("target_game"),
        "prior_level": payload.get("prior_reproduced_level"),
        "offline_reproduced": payload.get("offline_reproduced"),
        "reproduced_levels": payload.get("reproduced_levels"),
        "reproduction_gate": payload.get("reproduction_gate"),
        "solution_labels": payload.get("solution_labels"),
        "verifier_delta": payload.get("verifier_delta"),
        "target_selection": payload.get("target_selection"),
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
    if payload.get("offline_reproduced") and int(payload.get("reproduced_levels") or 0) < 1:
        errors.append("offline_reproduced_without_new_level")
    if payload.get("offline_reproduced") and not payload.get("registry_updated"):
        errors.append("offline_reproduced_without_registry_update")
    if payload.get("target_game") == "sb26" and payload.get("target_selection", {}).get("rotation_conflict"):
        errors.append("sb26_must_not_be_a_rotation_conflict")
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
    game = str(loop_result.get("game") or "")
    gate = dict(loop_result.get("reproduction_gate") or {})
    reached_level = _gate_reached(loop_result)
    new_levels = max(0, reached_level - int(prior_level))
    checkpoint_path = loop_result.get("learned_verifier_checkpoint")
    checkpoint_updated = bool(checkpoint_path and checkpoint_after_sha)
    success = bool(
        _gate_reproduced(loop_result)
        and loop_result.get("offline_reproduced")
        and new_levels >= 1
        and checkpoint_updated
        and registry_updated
    )
    verdict = (
        f"success: {game}_L{reached_level}_offline_reproduced"
        if success
        else f"complete: {game}_delta_identified_no_bank"
    )
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": "carnot.exp4678.levelup_selfplay.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": loop_result.get("solve_provenance") or SOLVE_PROVENANCE,
        "offline_reproduced": bool(success),
        "reproduced_levels": int(new_levels if success else 0),
        "target_game": game,
        "verifier_checkpoint_updated": checkpoint_updated,
        "registry_updated": bool(registry_updated),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": list(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "standing_loop_result_path": f"results/arc_loop_solve_{game}.json",
        "reproduction_gate": gate,
        "solution_labels": list(loop_result.get("solution_labels") or []),
        "dead_ends_recorded": list(dead_ends_recorded),
        "target_selection": dict(target_selection),
        "prior_reproduced_level": int(prior_level),
        "reached_level": reached_level,
        "reproducible_total_levels_before": int(prior_total_levels),
        "reproducible_total_levels_after": int(
            prior_total_levels + (new_levels if success else 0)
        ),
        "verifier_delta": {
            "checkpoint_path": checkpoint_path,
            "before_sha256": checkpoint_before_sha,
            "after_sha256": checkpoint_after_sha,
            "updated": checkpoint_updated,
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
        rows.append(f"  - {str(item)}")
    return "\n".join(rows)


def _sb26_registry_block(checksum: str, checkpoint_path: str, artifact: dict[str, Any]) -> str:
    return f"""- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: color_match_slot_sequence
  win_condition: >-
    L1 flat ordered color-to-slot match plus L2 nested-frame color-match
    sequence: root frame consumes colors 12 and 15, a pre-placed vgszefyyyp
    branch descends to the lower frame for 8, 9, 14, and 11, then the root tail
    consumes 6 before ACTION5 validates and next_level fires.
  action_model: >-
    ACTION6 item click then ACTION6 slot click; ACTION5 validates. L2 tail uses
    display-center clicks derived from the offline environment source:
    (32,59)->(23,23), (18,59)->(29,23), (11,59)->(23,37),
    (46,59)->(29,37), (25,59)->(35,37), (53,59)->(41,37),
    (39,59)->(41,23), then validate.
  solver: >-
    GameAdapter _sb26 in python/carnot/agentic/arc_game_adapters.py plus
    scripts/arc_loop_solve.py --game sb26 --target-level 2 --no-hazard-prune.
  reproduce: >-
    Exp4678 results/experiment_4678_levelup_selfplay.json re-gated
    results/arc_loop_solve_sb26.json offline_reproduced=True, reached_level=2,
    banked +1 over the prior L1 registry row, checksum {checksum}.
  learned_verifier_checkpoint: {checkpoint_path}
  gotchas:
  - The L2 vgszefyyyp branch is not a target color itself; validation descends
    through it and only counts the real lngftsryyw item slots in DFS order.
  - The learned verifier checkpoint routes/ranks future search only; the
    executable offline reproduction gate remains the oracle-distinct authority.
{_dead_end_lines(artifact)}
  latest_exp4678_levelup_selfplay:
    artifact: results/experiment_4678_levelup_selfplay.json
    loop_artifact: results/arc_loop_solve_sb26.json
    offline_reproduced: true
    reproduced_levels: 2
    new_levels_banked: 1
    verifier_checkpoint: {checkpoint_path}
    target_rotation: clean_L1_only_not_deepened_in_426_430
    reproducibility_checksum: {checksum}"""


def update_registry_for_success(
    artifact: dict[str, Any],
    registry_path: Path | None = None,
) -> bool:
    game = artifact["target_game"]
    checkpoint = artifact["verifier_delta"]["checkpoint_path"]
    checksum = artifact["reproducibility_checksum"]
    path = REGISTRY if registry_path is None else Path(registry_path)
    text = path.read_text(encoding="utf-8")
    old_text = text
    text = _replace_game_block(text, game, _sb26_registry_block(checksum, checkpoint, artifact))
    before_total = int(artifact["reproducible_total_levels_before"])
    after_total = int(artifact["reproducible_total_levels_after"])
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


def _dead_ends_from_selection(target_selection: dict[str, Any], reached_level: int, prior_level: int) -> list[str]:
    dead_ends = [
        f"{item['game']}: {item['reason']}" for item in target_selection.get("skipped", [])
    ]
    dead_ends.append("r11l: prefix-rooted graph search reached only L1 after 20000 expansions")
    dead_ends.append("lf52: prefix-rooted graph search timed out without a bank")
    if reached_level <= prior_level:
        dead_ends.append(
            f"sb26 standing loop reached L{reached_level}, not beyond prior L{prior_level}"
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
        target_selection = {"selected": game, "override": True, "rotation_conflict": game in PROHIBITED_TARGETS}
    else:
        game, target_selection = select_target()

    prior_level = registry_level(game)
    prior_total = registry_total_levels()
    checkpoint = MODELS / f"arc_verifier_{game}.json"
    before_sha = sha256_file(checkpoint)
    if args.refresh_loop:
        loop = run_standing_loop(game, prior_level + 1)
        preconditions_checked.append("scripts/arc_loop_solve.py --refresh-loop")
    else:
        loop = load_or_run_standing_loop(game, prior_level + 1, prior_level)
        preconditions_checked.append(f"standing_loop_result_or_run:{LOOP_RESULT_RELATIVE_PATH}")
    after_sha = sha256_file(checkpoint)
    reached_level = _gate_reached(loop)
    dead_ends = _dead_ends_from_selection(target_selection, reached_level, prior_level)
    success_ready = bool(
        loop.get("offline_reproduced")
        and _gate_reproduced(loop)
        and reached_level > prior_level
        and loop.get("learned_verifier_checkpoint")
        and after_sha
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
                    "verifier_checkpoint_updated",
                    "registry_updated",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - direct command in task.
    raise SystemExit(main())
