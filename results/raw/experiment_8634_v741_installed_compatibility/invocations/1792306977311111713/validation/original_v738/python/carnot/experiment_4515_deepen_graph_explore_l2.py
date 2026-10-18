"""Experiment 4515: graph-explore m0r0 L1->L2 deepening.

Spec refs: REQ-ARC-WMTE-4515, SCENARIO-ARC-WMTE-4515.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


RESULT_RELATIVE_PATH = "results/experiment_4515_deepen_graph_explore_l2.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-ARC-WMTE-4515", "SCENARIO-ARC-WMTE-4515"]
TARGET_GAME = "m0r0"
PRIOR_REPRODUCED_LEVELS = 1
TARGET_LEVEL = 2
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; e.g. success: <game>_L2_offline_reproduced OR "
        "complete: <game>_l2_honest_residual."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade reproduce, no LLM load "
        "(1s floor)."
    ),
    "offline_reproduced": (
        "a solve not reproducible offline is wasted effort -- only reproduced levels count "
        "(ARC Solve Reproducibility)."
    ),
    "reproduced_levels": (
        "the banked level count for the game (must be 2 for a successful deepen)."
    ),
    "target_game": (
        "the deepen target -- a NEW graph-explore game, not the HUD-register-stall ka59/ar25."
    ),
    "reproducibility_checksum": (
        "content-addressed hash of the reproduced replay -- the integrity gate against count inflation."
    ),
    "preconditions_checked": (
        "records resources verified; pre-empts missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "offline_reproduced",
    "reproduced_levels",
    "target_game",
    "reproducibility_checksum",
    "adapter_registered",
    "reproduction_gate",
    "solution_labels",
    "residual_blockers",
)
HUD_REGISTER_STALL_GAMES = {"ka59", "ar25"}


def ensure_preconditions_ready(preconditions: Mapping[str, Any]) -> None:
    if not preconditions.get("offline_arcade_import_smoke"):
        raise RuntimeError("blocked_offline_arcade_import_smoke")


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    Path(root)
    preconditions: dict[str, Any] = {
        "offline_arcade_import_smoke": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade_import_smoke"] = True
    except Exception as exc:
        preconditions["offline_arcade_error"] = repr(exc)
    return preconditions


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    replay_payload = {
        "target_game": payload.get("target_game"),
        "claimed_total_level": payload.get("claimed_total_level"),
        "adapter_branch_mode": payload.get("adapter_branch_mode"),
        "l1_prefix_labels": payload.get("l1_prefix_labels"),
        "l2_extension_labels": payload.get("l2_extension_labels"),
        "solution_labels": payload.get("solution_labels"),
        "reproduction_gate": payload.get("reproduction_gate"),
    }
    encoded = json.dumps(replay_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("experiment") != "experiment_4515_deepen_graph_explore_l2":
        errors.append("experiment must equal experiment_4515_deepen_graph_explore_l2")
    if artifact.get("schema") != "carnot.graph_explore_deepen_l2_4515.v1":
        errors.append("schema must equal carnot.graph_explore_deepen_l2_4515.v1")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs must match REQ-ARC-WMTE-4515")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal verifier_ensemble_against_cached_candidates")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    elif not artifact["preconditions_checked"].get("offline_arcade_import_smoke"):
        errors.append("preconditions_checked must record offline_arcade_import_smoke=true")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if not isinstance(artifact.get("reproduction_gate"), Mapping):
        errors.append("reproduction_gate must be a mapping")
    if not isinstance(artifact.get("solution_labels"), list):
        errors.append("solution_labels must be a list")
    if not isinstance(artifact.get("l1_prefix_labels"), list):
        errors.append("l1_prefix_labels must be a list")
    if not isinstance(artifact.get("l2_extension_labels"), list):
        errors.append("l2_extension_labels must be a list")
    target_game = str(artifact.get("target_game", ""))
    if target_game in HUD_REGISTER_STALL_GAMES:
        errors.append("target_game must not be a HUD-register-stall game ka59/ar25")
    if target_game != TARGET_GAME:
        errors.append("target_game must equal m0r0")
    checksum = artifact.get("reproducibility_checksum")
    expected_checksum = _checksum_payload(artifact)
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be a sha256 hex digest")
    elif checksum != expected_checksum:
        errors.append("reproducibility_checksum must match checksum of reproduced replay")
    if verdict and str(verdict).startswith(("success:", "success_")):
        gate = artifact.get("reproduction_gate", {})
        gate_reached = int(gate.get("reached_level", 0) or 0) if isinstance(gate, Mapping) else 0
        if artifact.get("adapter_registered") is not True:
            errors.append("success artifact requires adapter_registered=true")
        if artifact.get("adapter_branch_mode") != "fresh_env":
            errors.append("success artifact requires adapter_branch_mode=fresh_env")
        if artifact.get("offline_reproduced") is not True or int(artifact.get("reproduced_levels", 0)) != 2:
            errors.append("success artifact requires offline_reproduced=true and reproduced_levels=2")
        if not isinstance(gate, Mapping) or gate.get("reproduced") is not True or gate_reached < TARGET_LEVEL:
            errors.append("success artifact requires reproduction gate reached_level >= 2")
    return errors


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    target_game: str,
    adapter_registered: bool,
    solution_labels: Sequence[str],
    l1_prefix_labels: Sequence[str],
    solve_reached_level: int,
    reproduction_gate: Mapping[str, Any],
    depth_cap: int,
    states_expanded: int,
    tests_pass: bool,
    adapter_branch_mode: str = "fresh_env",
) -> dict[str, Any]:
    ensure_preconditions_ready(preconditions_checked)
    l1_prefix = list(l1_prefix_labels)
    extension = list(solution_labels)
    full_solution = l1_prefix + extension
    gate_reached_level = int(reproduction_gate.get("reached_level", 0) or 0)
    offline_reproduced = bool(reproduction_gate.get("reproduced", False)) and gate_reached_level >= TARGET_LEVEL
    residuals: list[str] = []
    if not adapter_registered:
        residuals.append(f"{target_game}_adapter_not_registered")
    if int(solve_reached_level) < TARGET_LEVEL:
        residuals.append(f"{target_game}_solver_reached_level_{int(solve_reached_level)}")
    if not offline_reproduced:
        residuals.append(str(reproduction_gate.get("residual") or f"{target_game}_l2_not_reproduced"))
    verdict = (
        f"success: {target_game}_L2_offline_reproduced"
        if offline_reproduced and adapter_registered
        else f"complete: {target_game}_l2_honest_residual"
    )
    payload: dict[str, Any] = {
        "experiment": "experiment_4515_deepen_graph_explore_l2",
        "schema": "carnot.graph_explore_deepen_l2_4515.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "preconditions_checked": dict(preconditions_checked),
        "target_game": str(target_game),
        "adapter_registered": bool(adapter_registered),
        "adapter_branch_mode": str(adapter_branch_mode),
        "prior_reproduced_levels": PRIOR_REPRODUCED_LEVELS,
        "claimed_total_level": TARGET_LEVEL,
        "solve_reached_level": int(solve_reached_level),
        "offline_reproduced": bool(offline_reproduced),
        "reproduced_levels": int(TARGET_LEVEL if offline_reproduced else PRIOR_REPRODUCED_LEVELS),
        "total_reproduced_levels": int(gate_reached_level),
        "reproduction_gate": dict(reproduction_gate),
        "l1_prefix_labels": l1_prefix,
        "l2_extension_labels": extension,
        "solution_labels": full_solution,
        "depth_cap": int(depth_cap),
        "states_expanded": int(states_expanded),
        "residual_blockers": residuals,
        "tests_pass": bool(tests_pass),
        "schema_errors": [],
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    payload["schema_errors"] = artifact_schema_errors(payload)
    return payload


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _prior_l1_labels(root: Path | str = REPO_ROOT) -> list[str]:  # pragma: no cover
    trajectory_path = Path(root) / "results" / "arc_explore_trajectory_m0r0.json"
    data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    return [
        json.dumps({"action": int(row["action"])}, sort_keys=True, separators=(",", ":"))
        for row in data.get("trajectory", [])
    ]


def _real_solver_runner(  # pragma: no cover
    game: str,
    adapter: Any,
    target_level: int,
    depth_cap: int,
) -> tuple[list[str], list[str], int, int]:
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        game_id=game,
        action_labels=adapter.action_labels,
        apply=adapter.apply,
        state_key=adapter.state_key,
        warmup_label=adapter.warmup_label,
        verifier=adapter.hand_verifier,
        branch_mode=adapter.branch_mode,
        max_nodes=50000,
    )
    prefix = _prior_l1_labels()
    extension, states = solver.solve_level(
        env,
        start_level=PRIOR_REPRODUCED_LEVELS,
        prefix=prefix,
        depth_cap=depth_cap,
    )
    extension = list(extension or [])
    frame = solver._replay(env, prefix + extension)
    reached = kit.frame_level(frame)
    return prefix, extension, int(reached), int(states)


def _real_reproduction_runner(  # pragma: no cover
    game: str,
    labels: Sequence[str],
    apply_fn: Callable[..., Any],
    *,
    warmup_label: str | None,
    claimed_level: int,
) -> Mapping[str, Any]:
    from carnot.agentic import arc_solver_kit as kit

    return kit.reproduce(
        game,
        labels,
        apply_fn,
        warmup_label=warmup_label,
        claimed_level=claimed_level,
    )


def _adapter_lookup(game: str) -> Any:  # pragma: no cover
    from carnot.agentic.arc_game_adapters import get_adapter

    return get_adapter(game)


def run_experiment(
    *,
    root: Path | str = REPO_ROOT,
    adapter_lookup: Callable[[str], Any] | None = None,
    solver_runner: Callable[[str, Any, int, int], tuple[Sequence[str], Sequence[str], int, int]] | None = None,
    reproduction_runner: Callable[..., Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    tests_pass: bool = False,
) -> dict[str, Any]:
    checked = dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root)
    ensure_preconditions_ready(checked)
    lookup = adapter_lookup or _adapter_lookup
    adapter = lookup(TARGET_GAME)
    if adapter is None:
        artifact = build_artifact(
            preconditions_checked=checked,
            target_game=TARGET_GAME,
            adapter_registered=False,
            solution_labels=[],
            l1_prefix_labels=[],
            solve_reached_level=PRIOR_REPRODUCED_LEVELS,
            reproduction_gate={"game": TARGET_GAME, "reached_level": 1, "reproduced": False},
            depth_cap=0,
            states_expanded=0,
            tests_pass=tests_pass,
            adapter_branch_mode="missing",
        )
        write_artifact(artifact, root=root)
        return artifact

    depth_cap = int(getattr(adapter, "depth_caps", {}).get(TARGET_LEVEL, 80))
    solve = solver_runner or _real_solver_runner
    reproduce = reproduction_runner or _real_reproduction_runner
    prefix, extension, reached, states = solve(TARGET_GAME, adapter, TARGET_LEVEL, depth_cap)
    full_solution = list(prefix) + list(extension)
    gate = reproduce(
        TARGET_GAME,
        full_solution,
        adapter.apply,
        warmup_label=getattr(adapter, "warmup_label", None),
        claimed_level=TARGET_LEVEL,
    )
    artifact = build_artifact(
        preconditions_checked=checked,
        target_game=TARGET_GAME,
        adapter_registered=True,
        solution_labels=list(extension),
        l1_prefix_labels=list(prefix),
        solve_reached_level=int(reached),
        reproduction_gate=gate,
        depth_cap=depth_cap,
        states_expanded=int(states),
        tests_pass=tests_pass,
        adapter_branch_mode=str(getattr(adapter, "branch_mode", "unknown")),
    )
    write_artifact(artifact, root=root)
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run_experiment(preconditions_checked=check_preconditions(), tests_pass=False)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
