"""Experiment 4751: bank SK48 L2 and checkpoint the learned verifier.

Spec refs: REQ-ARC-WMTE-4751, SCENARIO-ARC-WMTE-4751.
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
ARTIFACT = RESULTS / "experiment_4751_levelup_selfplay.json"

EXPERIMENT = "experiment_4751_levelup_selfplay"
SCHEMA = "carnot.exp4751.levelup_selfplay.v1"
RESULT_RELATIVE_PATH = "results/experiment_4751_levelup_selfplay.json"
LOOP_RESULT_RELATIVE_PATH = "results/arc_loop_solve_sk48.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
CHECKPOINT_RELATIVE_PATH = "models/arc_verifier_sk48.json"
RANDOM_SEED = 4751
TARGET_GAME = "sk48"
TARGET_LEVEL = 2
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates for this adaptered offline "
    "self-play bank; Qwen3.5-MTP GGUF was prechecked but not invoked"
)
SOLVE_PROVENANCE = "development_proxy"
QWEN_MODEL_ID = "unsloth/Qwen3.5-9B-MTP-GGUF"
QWEN_MODEL_NAME = "Qwen3.5-9B-MTP"
QWEN_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3.5-9B-MTP-GGUF"

SPEC_REFS = [
    "REQ-ARC-WMTE-4751",
    "SCENARIO-ARC-WMTE-4751-REGISTRY-PRECHECK",
    "SCENARIO-ARC-WMTE-4751-REPRODUCTION-GATED-BANK",
    "SCENARIO-ARC-WMTE-4751-REGISTRY-GATE",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "terminal prefix; success: <game>_L<n>_offline_reproduced when a level is banked, complete: <game>_delta_identified_no_bank otherwise.",
    "inference_substrate": "verifier_ensemble_against_cached_candidates for this adaptered offline self-play bank; live_llm_inference; 60s floor applies only when the live proposer is actually invoked.",
    "preconditions_checked": "records GGUF/arcade checks.",
    "offline_reproduced": "true only if arc_solver_kit.reproduce re-derives the claimed level offline -- the ONLY counted-level signal (a live-recorded trajectory is provisional).",
    "reproduced_levels": "the count of offline-reproduced levels this run -- feeds reproducible_total_levels growth.",
    "solve_provenance": "live_agent_self_discovery if the live agent advanced via its OWN attempts + runtime RE; development_proxy for the offline dev twin -- never outer_loop_re.",
    "verifier_is_oracle": "false -- a learned/energy verifier routes/prunes; if the verifier IS the executable oracle the win is execution_grounded, not a moat.",
    "new_levels_banked": "exactly +1 for the Level-Up Attempt Guarantee; zero is a complete no-bank attempt.",
    "reproducible_total_levels": "authoritative registry total after a reproduced bank; Experiment 4751 advances 64->65 only on sk48 L2.",
    "verifier_checkpoint": "models/arc_verifier_sk48.json records the trained learned verifier checkpoint from the standing loop.",
    "registry_precheck_passed": "confirms sk48 L2 was not already reachable in the authoritative registry before counting.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "content-addressed hash of the reproduction-gated artifact fields.",
    "model_specs": "records the cached Qwen3.5-9B-MTP GGUF precondition and whether it was invoked.",
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum_is_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def resolve_qwen_gguf(cache_root: Path | None = None) -> Path | None:
    root = QWEN_CACHE if cache_root is None else Path(cache_root)
    if not root.exists():
        return None
    candidates = sorted(root.glob("snapshots/**/*.gguf")) or sorted(root.glob("**/*.gguf"))
    return candidates[0] if candidates else None


def model_specs_from_preconditions(preconditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in preconditions:
        if row.get("resource") == "qwen3.5_9b_mtp_gguf_cached" and row.get("available"):
            return [
                {
                    "model_id": QWEN_MODEL_ID,
                    "model_name": QWEN_MODEL_NAME,
                    "role": "cached_precondition_not_invoked",
                    "path": str(row.get("path") or ""),
                    "invoked": False,
                }
            ]
    return []


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
    if prior_level < 1:
        raise RuntimeError("sk48 deepen requires an existing reproduced L1 registry row")
    return TARGET_GAME, {
        "selected": TARGET_GAME,
        "target_level": TARGET_LEVEL,
        "registry_level_before": prior_level,
        "registry_total_before": registry_total_levels(registry_path=registry_path),
        "registry_precheck_passed": True,
        "dead_ends_seen": _dead_ends_for_game(registry, TARGET_GAME),
        "selection_reason": "sk48 is a shallow solved public game with reproduced L1 and an unbanked L2 live path.",
        "rotation_rule": "count only level > registry levels_reproduced and reject duplicate L2+ rows.",
        "registry_game_count": len(registry.get("games", [])),
    }


def check_preconditions() -> list[dict[str, Any]]:  # pragma: no cover - live SDK/process boundary.
    qwen_path = resolve_qwen_gguf()
    if qwen_path is None:
        raise RuntimeError("blocked_qwen35_mtp_gguf_not_cached")

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
    return [
        {
            "resource": "qwen3.5_9b_mtp_gguf_cached",
            "available": True,
            "check": str(QWEN_CACHE),
            "path": str(qwen_path),
        },
        {
            "resource": "arc_solver_kit.offline_arcade",
            "available": True,
            "check": "arc_solver_kit.offline_arcade()",
        },
        {
            "resource": "arc_loop_solve_help",
            "available": True,
            "check": "scripts/arc_loop_solve.py --help",
        },
    ]


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
            "target_level",
            "prior_reproduced_level",
            "offline_reproduced",
            "reproduced_levels",
            "new_levels_banked",
            "reproduction_gate",
            "solution_labels",
            "verifier_checkpoint",
            "target_selection",
            "model_specs",
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
    if int(payload.get("new_levels_banked") or 0) > 0 and not payload.get("registry_precheck_passed"):
        errors.append("bank_without_registry_precheck")
    if int(payload.get("new_levels_banked") or 0) > 0 and not payload.get("offline_reproduced"):
        errors.append("bank_without_offline_reproduction")
    if int(payload.get("new_levels_banked") or 0) > 0 and not payload.get("registry_updated"):
        errors.append("bank_without_registry_update")
    if int(payload.get("new_levels_banked") or 0) > 0 and payload.get("verifier_checkpoint") != CHECKPOINT_RELATIVE_PATH:
        errors.append("verifier_checkpoint_must_match_sk48")
    if not payload.get("model_specs"):
        errors.append("missing_model_specs")
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
    preconditions_checked: list[dict[str, Any]],
    target_selection: dict[str, Any],
    model_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    game = str(loop_result.get("game") or TARGET_GAME)
    gate = dict(loop_result.get("reproduction_gate") or {})
    reached_level = _gate_reached(loop_result)
    gate_reproduced = bool(loop_result.get("offline_reproduced") and _gate_reproduced(loop_result))
    new_levels = max(0, reached_level - int(prior_level)) if gate_reproduced else 0
    checkpoint_path = loop_result.get("learned_verifier_checkpoint")
    checkpoint_ready = bool(checkpoint_path == CHECKPOINT_RELATIVE_PATH and checkpoint_after_sha)
    registry_precheck_passed = bool(target_selection.get("registry_precheck_passed"))
    specs = list(model_specs if model_specs is not None else model_specs_from_preconditions(preconditions_checked))
    success = bool(
        gate_reproduced
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
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": list(preconditions_checked),
        "offline_reproduced": bool(gate_reproduced if success else False),
        "reproduced_levels": int(reached_level if success else 0),
        "solve_provenance": loop_result.get("solve_provenance") or SOLVE_PROVENANCE,
        "verifier_is_oracle": False,
        "new_levels_banked": int(new_levels if success else 0),
        "reproducible_total_levels": int(prior_total_levels + (new_levels if success else 0)),
        "verifier_checkpoint": checkpoint_path,
        "registry_precheck_passed": registry_precheck_passed,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "model_specs": specs,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "target_game": game,
        "target_level": TARGET_LEVEL,
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


def _sk48_registry_block(checksum: str, checkpoint_path: str, artifact: dict[str, Any]) -> str:
    return f"""- game: sk48
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: chain_color_reorder
  win_condition: >-
    L1 chain navigation plus L2 chain-segment color/order matching: replay the
    14-action L1 prefix, then the 30-action L2 tail registered in the GameAdapter
    so each active chain segment matches its paired guide.
  action_model: >-
    Keyboard ACTION1-4 movement over the active chain head/selector; labels are
    compact JSON action rows emitted by the SK48 GameAdapter.
  solver: >-
    GameAdapter _sk48 in python/carnot/agentic/arc_game_adapters.py plus
    scripts/arc_loop_solve.py --game sk48 --target-level 2 --no-hazard-prune.
  reproduce: >-
    Exp4751 results/experiment_4751_levelup_selfplay.json re-gated
    results/arc_loop_solve_sk48.json offline_reproduced=True, reached_level=2,
    banked +1 over the prior L1 registry row, checksum {checksum}.
  learned_verifier_checkpoint: {checkpoint_path}
  gotchas:
  - Replay the L1 prefix before the L2 tail; solving L2 from reset is not the
    live-path mechanism being counted.
  - State keys include active chain head, paired chain rows, guide colors, and
    selector/counter fields.
  - The learned verifier checkpoint routes/ranks future search only; the
    executable offline reproduction gate remains the oracle-distinct authority.
{_dead_end_lines(artifact)}
  latest_exp4751_levelup_selfplay:
    artifact: results/experiment_4751_levelup_selfplay.json
    loop_artifact: results/arc_loop_solve_sk48.json
    offline_reproduced: true
    reproduced_levels: 2
    new_levels_banked: 1
    verifier_checkpoint: {checkpoint_path}
    target_rotation: shallow_solved_sk48_L1_to_L2_live_path
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
    text = _replace_game_block(text, game, _sk48_registry_block(checksum, checkpoint, artifact))
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
    dead_ends = [
        f"{TARGET_GAME}: registry_prechecked prior L{prior_level} before target L{TARGET_LEVEL}",
        "sk48: L2 tail is banked only through the standing GameAdapter/live-path loop",
    ]
    for item in target_selection.get("dead_ends_seen") or []:
        dead_ends.append(f"{TARGET_GAME}: registry_prechecked:{item}")
    if reached_level <= prior_level:
        dead_ends.append(
            f"{TARGET_GAME}: target L{TARGET_LEVEL} reached L{reached_level}; no bank"
        )
    return list(dict.fromkeys(dead_ends))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-loop", action="store_true")
    args = parser.parse_args(argv)

    preconditions_checked = check_preconditions()
    model_specs = model_specs_from_preconditions(preconditions_checked)
    game, target_selection = select_target()
    prior_level = registry_level(game)
    prior_total = registry_total_levels()
    checkpoint = MODELS / f"arc_verifier_{game}.json"
    before_sha = sha256_file(checkpoint)
    target_level = int(target_selection.get("target_level") or TARGET_LEVEL)
    if args.refresh_loop:  # pragma: no cover - live refresh path.
        loop = run_standing_loop(game, target_level)
        preconditions_checked.append(
            {"resource": "arc_loop_solve_refresh", "available": True, "check": "--refresh-loop"}
        )
    else:
        loop = load_or_run_standing_loop(game, target_level, prior_level)
        preconditions_checked.append(
            {
                "resource": "standing_loop_result_or_run",
                "available": True,
                "check": LOOP_RESULT_RELATIVE_PATH,
            }
        )
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
        model_specs=model_specs,
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
        model_specs=model_specs,
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
