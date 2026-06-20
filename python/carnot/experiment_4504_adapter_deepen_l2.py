"""Experiment 4504: adapter-routed cd82 L1->L2 deepening.

Spec refs: REQ-ARC-WMTE-4504, SCENARIO-ARC-WMTE-4504.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot import experiment_4494_adapter_deepen_l2 as base


RESULT_RELATIVE_PATH = "results/experiment_4504_adapter_deepen_l2.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = base.INFERENCE_SUBSTRATE
SPEC_REFS = ["REQ-ARC-WMTE-4504", "SCENARIO-ARC-WMTE-4504"]
TARGET_GAME = "cd82"
PRIOR_REPRODUCED_LEVELS = base.PRIOR_REPRODUCED_LEVELS
TARGET_LEVEL = base.TARGET_LEVEL
TERMINAL_PREFIXES = base.TERMINAL_PREFIXES
FIELD_PRINCIPLES = dict(base.FIELD_PRINCIPLES)
REQUIRED_ARTIFACT_FIELDS = base.REQUIRED_ARTIFACT_FIELDS


def ensure_preconditions_ready(preconditions: Mapping[str, Any]) -> None:
    base.ensure_preconditions_ready(preconditions)


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    return base.check_preconditions(root)


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = base.artifact_schema_errors(artifact)
    if artifact.get("experiment") != "experiment_4504_adapter_deepen_l2":
        errors.append("experiment must equal experiment_4504_adapter_deepen_l2")
    if artifact.get("schema") != "carnot.adapter_deepen_l2_4504.v1":
        errors.append("schema must equal carnot.adapter_deepen_l2_4504.v1")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs must match REQ-ARC-WMTE-4504")
    if artifact.get("target_game") != TARGET_GAME:
        errors.append("target_game must equal cd82")
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
    payload = base.build_artifact(
        preconditions_checked=preconditions_checked,
        target_game=target_game,
        adapter_registered=adapter_registered,
        solution_labels=solution_labels,
        solve_reached_level=solve_reached_level,
        reproduction_gate=reproduction_gate,
        depth_cap=depth_cap,
        states_expanded=states_expanded,
        tests_pass=tests_pass,
        adapter_branch_mode=adapter_branch_mode,
    )
    payload.update(
        {
            "experiment": "experiment_4504_adapter_deepen_l2",
            "schema": "carnot.adapter_deepen_l2_4504.v1",
            "spec_refs": list(SPEC_REFS),
        }
    )
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
    return base._real_solver_runner(game, adapter, target_level, depth_cap)


def _real_reproduction_runner(  # pragma: no cover
    game: str,
    labels: Sequence[str],
    apply_fn: Callable[..., Any],
    *,
    warmup_label: str | None,
    claimed_level: int,
) -> Mapping[str, Any]:
    return base._real_reproduction_runner(
        game,
        labels,
        apply_fn,
        warmup_label=warmup_label,
        claimed_level=claimed_level,
    )


def _adapter_lookup(game: str) -> Any:  # pragma: no cover
    return base._adapter_lookup(game)


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
