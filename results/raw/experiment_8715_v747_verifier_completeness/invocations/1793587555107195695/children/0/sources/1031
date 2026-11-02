"""Experiment 4630: bank ls20 L2 through the standing ARC self-play loop.

Spec refs: REQ-ARC-WMTE-4630, SCENARIO-ARC-WMTE-4630.
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

from carnot.agentic.arc_game_adapters import (
    LS20_L1_LABELS,
    LS20_L2_SOLUTION_LABELS,
    LS20_L2_TAIL_LABELS,
)


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
MODELS = REPO / "models"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4630_levelup_selfplay.json"

EXPERIMENT = "experiment_4630_levelup_selfplay"
RESULT_RELATIVE_PATH = "results/experiment_4630_levelup_selfplay.json"
LOOP_RESULT_RELATIVE_PATH = "results/arc_loop_solve_ls20.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
CHECKPOINT_RELATIVE_PATH = "models/arc_verifier_ls20.json"
RANDOM_SEED = 4630
SPEC_REFS = ["REQ-ARC-WMTE-4630", "SCENARIO-ARC-WMTE-4630"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "development_proxy"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank).",
    "inference_substrate": "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged.",
    "verifier_is_oracle": "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check.",
    "solve_provenance": "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game.",
    "offline_reproduced": "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels.",
    "reproduced_levels": "the integer new-level count banked this task (>=1 satisfies the level-up guarantee).",
    "target_game": "the rotated game attempted -- traceable to the rotation discipline (a clean-nav game not deepened in .421-.426, not sk48/dc22/ka59/wa30).",
    "verifier_checkpoint_updated": "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone).",
    "registry_updated": "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

PREFERRED_TARGETS = ("ls20", "r11l", "g50t", "bp35", "re86", "lf52")
FAILED_OR_DEEPENED_421_426 = {"sk48", "dc22", "ft09", "ar25", "m0r0", "cn04"}
HIDDEN_OR_STATE_BOUND = {"ka59", "wa30"}
NO_GROUNDED_NEXT_DELTA = {"cd82", "sp80", "su15"}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry_data(registry_path: Path = REGISTRY) -> dict[str, Any]:
    return yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}


def _game_entry(registry: dict[str, Any], game: str) -> dict[str, Any]:
    for entry in registry.get("games", []):
        if entry.get("game") == game:
            return entry
    return {}


def registry_level(game: str, registry_path: Path = REGISTRY) -> int:
    entry = _game_entry(_registry_data(registry_path), game)
    try:
        return int(entry.get("levels_reproduced") or 0)
    except (TypeError, ValueError):
        return 0


def registry_total_levels(registry_path: Path = REGISTRY) -> int:
    try:
        return int(_registry_data(registry_path).get("reproducible_total_levels") or 0)
    except (TypeError, ValueError):
        return 0


def _skip_reasons(game: str, entry: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if game in FAILED_OR_DEEPENED_421_426:
        reasons.append("failed_or_deepened_in_421_426_rotation_window")
    if game in HIDDEN_OR_STATE_BOUND:
        reasons.append("hidden_or_state_bound_dead_end")
    if game in NO_GROUNDED_NEXT_DELTA:
        reasons.append("no_grounded_next_delta")
    try:
        reproduced = int(entry.get("levels_reproduced") or 0)
    except (TypeError, ValueError):
        reproduced = 0
    if reproduced != 1:
        reasons.append("not_a_shallow_L1_target")
    return reasons


def select_target(
    preferred: tuple[str, ...] = PREFERRED_TARGETS, registry_path: Path = REGISTRY
) -> tuple[str, dict[str, Any]]:
    registry = _registry_data(registry_path)
    skipped: list[dict[str, str]] = []
    for game in preferred:
        entry = _game_entry(registry, game)
        reasons = _skip_reasons(game, entry)
        if reasons:
            skipped.append({"game": game, "reason": "; ".join(reasons)})
            continue
        return game, {
            "preferred_targets": list(preferred),
            "selected": game,
            "skipped": skipped,
            "rotation_rule": "clean_nav_L1_not_deepened_or_failed_in_421_426_not_sk48_dc22_ka59_wa30",
        }
    raise RuntimeError(f"no eligible target in {preferred!r}")


def run_standing_loop(game: str, target_level: int) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/arc_loop_solve.py",
        "--game",
        game,
        "--target-level",
        str(target_level),
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


def _stable_checksum(payload: dict[str, Any]) -> str:
    checksum_payload = {
        "target_game": payload.get("target_game"),
        "prior_level": payload.get("prior_reproduced_level"),
        "offline_reproduced": payload.get("offline_reproduced"),
        "reproduced_levels": payload.get("reproduced_levels"),
        "reproduction_gate": payload.get("reproduction_gate"),
        "solution_labels": payload.get("solution_labels"),
        "verifier_delta": payload.get("verifier_delta"),
    }
    raw = json.dumps(checksum_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
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
    checksum = payload.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
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
    target_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    game = str(loop_result.get("game") or "")
    gate = dict(loop_result.get("reproduction_gate") or {})
    reached_level = int(gate.get("reached_level") or loop_result.get("reached_level") or 0)
    gate_reproduced = bool(gate.get("reproduced", loop_result.get("offline_reproduced")))
    checkpoint_path = loop_result.get("learned_verifier_checkpoint")
    checkpoint_updated = bool(checkpoint_path and checkpoint_after_sha)
    new_levels = max(0, reached_level - int(prior_level))
    success = bool(gate_reproduced and new_levels >= 1 and checkpoint_updated and registry_updated)
    verdict = (
        f"success: {game}_L{reached_level}_offline_reproduced"
        if success
        else f"complete: {game}_delta_identified_no_bank"
    )
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
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
        "target_selection": target_selection or {},
        "prior_reproduced_level": int(prior_level),
        "reached_level": reached_level,
        "reproducible_total_levels_before": int(prior_total_levels),
        "reproducible_total_levels_after": int(prior_total_levels + (new_levels if success else 0)),
        "verifier_delta": {
            "checkpoint_path": checkpoint_path,
            "before_sha256": checkpoint_before_sha,
            "after_sha256": checkpoint_after_sha,
            "updated": checkpoint_updated,
            "sha_changed": bool(checkpoint_after_sha and checkpoint_after_sha != checkpoint_before_sha),
            "positive_trace_steps": len(loop_result.get("solution_labels") or []),
            "negative_trace_notes": list(dead_ends_recorded),
        },
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


def _ls20_registry_block(checksum: str, checkpoint_path: str) -> str:
    return f"""- game: ls20
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: clean_navigation_shape_color_rotation_step_counter
  win_condition: >-
    L1 first-solve navigation plus L2 clean-nav target match: enter each
    rjlbuycveu target only when the player shape/color/rotation tuple equals
    the paired kvynsvxbpi target tuple. L2 requires rotation index 3 at the
    target and two visible npxgalaybz counter resets before the step counter
    expires.
  action_model: >-
    Keyboard ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right over the
    five-pixel grid. L2 route: reach the rhsxkxzdjz rotation trigger, re-enter
    it until rotation index 3, collect the lower and upper npxgalaybz step
    resets, then enter the target.
  solver: >-
    GameAdapter _ls20 in python/carnot/agentic/arc_game_adapters.py +
    scripts/arc_loop_solve.py; results/arc_loop_solve_ls20.json 58-label L2
    gate.
  reproduce: >-
    Exp4630 results/experiment_4630_levelup_selfplay.json re-gated
    results/arc_loop_solve_ls20.json offline_reproduced=True,
    reached_level=2, banked +1 over the current L1 registry row, checksum
    {checksum}.
  learned_verifier_checkpoint: {checkpoint_path} trained by scripts/arc_loop_solve.py
    on the ls20 L1+L2 positive steps-to-go trace and the recorded rotation-skip
    dead-end notes.
  gotchas:
  - Read level progress from the returned frame; env._game.level_index is useful
    for adapter internals but not the reproduction claim.
  - L2 has a 42-step counter with two-step decrement, so the lower and upper
    npxgalaybz reset pickups are load-bearing.
  - The target blocks movement until shape/color/rotation all match.
  dead_ends:
  - sk48 is skipped because it was the .426 failed/deepened rotation target.
  - dc22 is skipped because it was the .425 failed deepen target.
  - ka59 and wa30 are skipped because their registry rows are hidden-state-bound.
  - cd82/sp80/su15 next deepens are skipped because no grounded next-level delta
    is recorded for this rotation.
  latest_exp4630_levelup_selfplay:
    artifact: results/experiment_4630_levelup_selfplay.json
    loop_artifact: results/arc_loop_solve_ls20.json
    offline_reproduced: true
    reproduced_levels: 2
    new_levels_banked: 1
    verifier_checkpoint: {checkpoint_path}
    reproducibility_checksum: {checksum}"""


def update_registry_for_success(
    artifact: dict[str, Any], registry_path: Path = REGISTRY
) -> bool:
    game = artifact["target_game"]
    checkpoint = artifact["verifier_delta"]["checkpoint_path"]
    checksum = artifact["reproducibility_checksum"]
    text = registry_path.read_text(encoding="utf-8")
    old_text = text
    text = _replace_game_block(text, game, _ls20_registry_block(checksum, checkpoint))
    before_total = int(artifact["reproducible_total_levels_before"])
    after_total = int(artifact["reproducible_total_levels_after"])
    text = re.sub(
        rf"(?m)^(reproducible_total_levels:\s*){before_total}\s*$",
        rf"\g<1>{after_total}",
        text,
        count=1,
    )
    if text != old_text:
        registry_path.write_text(text, encoding="utf-8")
    return text != old_text


def _write_artifact(payload: dict[str, Any], path: Path = ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game")
    args = parser.parse_args(argv)

    preconditions_checked = ["arc_solver_kit.offline_arcade()"]
    from carnot.agentic import arc_solver_kit as kit

    kit.offline_arcade()
    if args.game:
        game = args.game
        target_selection = {"selected": game, "override": True}
    else:
        game, target_selection = select_target()

    prior_level = registry_level(game)
    prior_total = registry_total_levels()
    checkpoint = MODELS / f"arc_verifier_{game}.json"
    before_sha = sha256_file(checkpoint)
    loop = run_standing_loop(game, prior_level + 1)
    after_sha = sha256_file(checkpoint)
    gate = dict(loop.get("reproduction_gate") or {})
    reached_level = int(gate.get("reached_level") or loop.get("reached_level") or 0)
    dead_ends = [
        f"{item['game']}: {item['reason']}" for item in target_selection.get("skipped", [])
    ]
    if reached_level <= prior_level:
        dead_ends.append(f"{game} standing loop reached L{reached_level}, not beyond prior L{prior_level}")

    success_ready = bool(
        loop.get("offline_reproduced")
        and gate.get("reproduced", True)
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


if __name__ == "__main__":  # pragma: no cover - exercised by the experiment command.
    raise SystemExit(main())
