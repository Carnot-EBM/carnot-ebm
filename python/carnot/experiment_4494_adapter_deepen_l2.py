"""Experiment 4494: adapter-routed ARC L1->L2 deepening.

Spec refs: REQ-ARC-WMTE-4496, SCENARIO-ARC-WMTE-4495.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


RESULT_RELATIVE_PATH = "results/experiment_4494_adapter_deepen_l2.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-ARC-WMTE-4496", "SCENARIO-ARC-WMTE-4495"]
TARGET_GAME = "cd82"
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
        "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "offline_reproduced",
    "reproduced_levels",
    "adapter_registered",
    "target_game",
    "reproduction_gate",
    "solution_labels",
    "residual_blockers",
)


def ensure_preconditions_ready(preconditions: Mapping[str, Any]) -> None:
    if not preconditions.get("offline_arcade_import_smoke"):
        raise RuntimeError("blocked_offline_arcade_import_smoke")
    if not preconditions.get("torch_import"):
        raise RuntimeError("blocked_torch_import")


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    Path(root)
    preconditions: dict[str, Any] = {
        "offline_arcade_import_smoke": False,
        "torch_import": False,
        "torch_version": "",
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade_import_smoke"] = True
    except Exception as exc:
        preconditions["offline_arcade_error"] = repr(exc)
    try:
        import torch

        preconditions["torch_import"] = True
        preconditions["torch_version"] = str(torch.__version__)
    except Exception as exc:
        preconditions["torch_error"] = repr(exc)
    return preconditions


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_reproduced_levels(reproduction_gate: Mapping[str, Any]) -> int:
    reached = int(reproduction_gate.get("reached_level", 0) or 0)
    return max(0, reached - PRIOR_REPRODUCED_LEVELS)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal verifier_ensemble_against_cached_candidates")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if not isinstance(artifact.get("reproduction_gate"), Mapping):
        errors.append("reproduction_gate must be a mapping")
    if not isinstance(artifact.get("solution_labels"), list):
        errors.append("solution_labels must be a list")
    if verdict and str(verdict).startswith(("success:", "success_")):
        if artifact.get("adapter_registered") is not True:
            errors.append("success artifact requires adapter_registered=true")
        if artifact.get("offline_reproduced") is not True or int(artifact.get("reproduced_levels", 0)) < 1:
            errors.append("success artifact requires offline_reproduced=true and reproduced_levels >= 1")
    return errors


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    target_game: str,
    adapter_registered: bool,
    solution_labels: Sequence[str],
    solve_reached_level: int,
    reproduction_gate: Mapping[str, Any],
    depth_cap: int,
    states_expanded: int,
    tests_pass: bool,
    adapter_branch_mode: str = "fresh_env",
) -> dict[str, Any]:
    ensure_preconditions_ready(preconditions_checked)
    gate_reached_level = int(reproduction_gate.get("reached_level", 0) or 0)
    new_levels = _new_reproduced_levels(reproduction_gate)
    offline_reproduced = (
        bool(reproduction_gate.get("reproduced", False))
        and gate_reached_level >= TARGET_LEVEL
        and new_levels >= 1
    )
    residuals: list[str] = []
    if not adapter_registered:
        residuals.append(f"{target_game}_adapter_not_registered")
    if int(solve_reached_level) < TARGET_LEVEL:
        residuals.append(f"{target_game}_solver_reached_level_{int(solve_reached_level)}")
    if not offline_reproduced:
        residuals.append(str(reproduction_gate.get("residual") or f"{target_game}_l2_not_reproduced"))
    verdict = (
        f"success: {target_game}_adapter_deepen_l2_offline_reproduced"
        if offline_reproduced and adapter_registered
        else f"complete: {target_game}_adapter_deepen_l2_honest_residual"
    )
    payload: dict[str, Any] = {
        "experiment": "experiment_4494_adapter_deepen_l2",
        "schema": "carnot.adapter_deepen_l2_4494.v1",
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
        "reproduced_levels": int(new_levels if offline_reproduced else 0),
        "total_reproduced_levels": gate_reached_level,
        "reproduction_gate": dict(reproduction_gate),
        "solution_labels": list(solution_labels),
        "depth_cap": int(depth_cap),
        "states_expanded": int(states_expanded),
        "residual_blockers": residuals,
        "tests_pass": bool(tests_pass),
    }
    payload["schema_errors"] = artifact_schema_errors(payload)
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _real_solver_runner(  # pragma: no cover
    game: str,
    adapter: Any,
    target_level: int,
    depth_cap: int,
) -> tuple[list[str], int, int]:
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
    solution, reached = solver.solve(env, target_level=target_level, depth_cap=depth_cap)
    return list(solution), int(reached), int(solver.last_states_expanded)


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
    solver_runner: Callable[[str, Any, int, int], tuple[Sequence[str], int, int]] | None = None,
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
    labels, reached, states = solve(TARGET_GAME, adapter, TARGET_LEVEL, depth_cap)
    gate = reproduce(
        TARGET_GAME,
        list(labels),
        adapter.apply,
        warmup_label=getattr(adapter, "warmup_label", None),
        claimed_level=TARGET_LEVEL,
    )
    artifact = build_artifact(
        preconditions_checked=checked,
        target_game=TARGET_GAME,
        adapter_registered=True,
        solution_labels=list(labels),
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
