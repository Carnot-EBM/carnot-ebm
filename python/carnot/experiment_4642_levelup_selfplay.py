"""Experiment 4642: bank ft09 L3 through the standing ARC self-play loop.

Spec refs: REQ-ARC-WMTE-4642, SCENARIO-ARC-WMTE-4642.
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
    FT09_L3_SOLUTION_LABELS,
    FT09_L3_TAIL_LABELS,
)


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
MODELS = REPO / "models"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
ARTIFACT = RESULTS / "experiment_4642_levelup_selfplay.json"

EXPERIMENT = "experiment_4642_levelup_selfplay"
RESULT_RELATIVE_PATH = "results/experiment_4642_levelup_selfplay.json"
LOOP_RESULT_RELATIVE_PATH = "results/arc_loop_solve_ft09.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
CHECKPOINT_RELATIVE_PATH = "models/arc_verifier_ft09.json"
RANDOM_SEED = 4642
SPEC_REFS = ["REQ-ARC-WMTE-4642", "SCENARIO-ARC-WMTE-4642"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "development_proxy"

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank).",
    "inference_substrate": "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged.",
    "verifier_is_oracle": "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check.",
    "solve_provenance": "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game.",
    "offline_reproduced": "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels.",
    "reproduced_levels": "the integer new-level count banked this task (>=1 satisfies the level-up guarantee).",
    "target_game": "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .422-.427, not ls20/sk48/dc22/ka59/wa30).",
    "verifier_checkpoint_updated": "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone).",
    "registry_updated": "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}

PREFERRED_DEAD_ENDS = (
    ("r11l", "adapter_free_l2_bounded_search_no_bank"),
    ("g50t", "adapter_free_l2_bounded_search_no_bank"),
    ("bp35", "adapter_free_l2_bounded_search_timeout"),
    ("m0r0", "standing_loop_repeated_prior_L2"),
    ("cn04", "standing_loop_repeated_prior_L2"),
)
PROHIBITED_TARGETS = ("ls20", "sk48", "dc22", "ka59", "wa30")


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


def select_target(
    fallback: str = "ft09",
    registry_path: Path = REGISTRY,
) -> tuple[str, dict[str, Any]]:
    registry = _registry_data(registry_path)
    fallback_level = registry_level(fallback, registry_path=registry_path)
    if fallback != "ft09" or fallback_level != 2:
        raise RuntimeError("ft09 fallback requires a reproduced L2 registry row")
    skipped = [{"game": game, "reason": reason} for game, reason in PREFERRED_DEAD_ENDS]
    return fallback, {
        "preferred_targets": [game for game, _reason in PREFERRED_DEAD_ENDS],
        "prohibited_targets": list(PROHIBITED_TARGETS),
        "selected": fallback,
        "fallback_exception": True,
        "exception_reason": (
            "preferred clean L1 targets and requested L2 alternatives were bounded "
            "dead-ends; ft09 L3 was the first offline-gateable adaptered bank"
        ),
        "skipped": skipped,
        "rotation_rule": ("skip_ls20_sk48_dc22_ka59_wa30; record fallback exception explicitly"),
        "registry_level_before": fallback_level,
        "registry_game_count": len(registry.get("games", [])),
    }


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
    target_selection: dict[str, Any],
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
        "target_selection": dict(target_selection),
        "prior_reproduced_level": int(prior_level),
        "reached_level": reached_level,
        "reproducible_total_levels_before": int(prior_total_levels),
        "reproducible_total_levels_after": int(prior_total_levels + (new_levels if success else 0)),
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


def _ft09_registry_block(checksum: str, checkpoint_path: str) -> str:
    return f"""- game: ft09
  reproducibility: reproduced
  levels_reproduced: 3
  mechanic_class: local_constraint_color_cycle
  win_condition: >-
    L1+L2 local-constraint color-cycle plus L3: every visible bsT 3x3
    predicate is satisfied by setting adjacent Hkx/NTi cells to either match or
    differ from the predicate center color before the move counter expires.
  action_model: >-
    ACTION6 click on display coordinates derived from the level camera. The L3
    delta clicks the forced Hkx cells at (28,20), (44,28), (28,36), (44,36),
    (20,44), (20,52), (28,52), (36,52), (20,4), (28,4), (36,4), (20,12),
    (12,20), and (12,28).
  solver: >-
    GameAdapter _ft09 in python/carnot/agentic/arc_game_adapters.py +
    scripts/arc_loop_solve.py; results/arc_loop_solve_ft09.json 25-label L3
    gate.
  reproduce: >-
    Exp4642 results/experiment_4642_levelup_selfplay.json re-gated
    results/arc_loop_solve_ft09.json offline_reproduced=True, reached_level=3,
    banked +1 over the current L2 registry row, checksum {checksum}.
  learned_verifier_checkpoint: {checkpoint_path} trained by scripts/arc_loop_solve.py
    on the ft09 L1+L2+L3 positive steps-to-go trace and the recorded preferred
    target dead-end notes.
  gotchas:
  - ft09 ACTION6 expects display coordinates; for L3 the camera maps display
    coordinates to grid coordinates by halving them.
  - The L3 click set is the binary color assignment forced by the four visible
    bsT predicates; using raw grid coordinates leaves most cells unchanged.
  - This is a fallback bank after preferred targets failed bounded attempts, so
    the rotation exception is recorded in the experiment artifact.
  dead_ends:
  - r11l adapter-free L2 bounded search did not bank.
  - g50t adapter-free L2 bounded search exhausted the bounded frontier.
  - bp35 adapter-free L2 bounded search timed out.
  - m0r0 and cn04 standing-loop L3 attempts repeated their prior L2 gates.
  - ls20, sk48, dc22, ka59, and wa30 are skipped by the operator's rotation
    exclusions for this milestone.
  latest_exp4642_levelup_selfplay:
    artifact: results/experiment_4642_levelup_selfplay.json
    loop_artifact: results/arc_loop_solve_ft09.json
    offline_reproduced: true
    reproduced_levels: 3
    new_levels_banked: 1
    verifier_checkpoint: {checkpoint_path}
    reproducibility_checksum: {checksum}"""


def update_registry_for_success(
    artifact: dict[str, Any],
    registry_path: Path = REGISTRY,
) -> bool:
    game = artifact["target_game"]
    checkpoint = artifact["verifier_delta"]["checkpoint_path"]
    checksum = artifact["reproducibility_checksum"]
    text = registry_path.read_text(encoding="utf-8")
    old_text = text
    text = _replace_game_block(text, game, _ft09_registry_block(checksum, checkpoint))
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
        dead_ends.append(
            f"{game} standing loop reached L{reached_level}, not beyond prior L{prior_level}"
        )

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
