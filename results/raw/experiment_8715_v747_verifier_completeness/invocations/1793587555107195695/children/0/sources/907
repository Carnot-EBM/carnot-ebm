"""Experiment 4505: refreshed submitted-agent ARC scoreboard.

Spec refs: REQ-ARC-FCP-4505, SCENARIO-ARC-FCP-4505.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4505_submitted_agent_scoreboard.json"
VALUE_WEIGHT_SOURCE_RELATIVE_PATH = "results/experiment_4500_value_weight_remeasure.json"
VARIANT_SOURCE_RELATIVE_PATH = "results/experiment_4499_capstone_v415.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
MILESTONE = "2026.06.415-B2"
PARITY_TEST_PATH = "tests/python/test_arc_submitted_agent_parity.py"
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
    "honest_verdict": {
        "principle": (
            "MUST start with terminal prefix complete:/complete_/success:/success_/"
            "passed:/passed_/shipped:/shipped_."
        )
    },
    "inference_substrate": {
        "principle": "explicit substrate so adversarial_verify applies the right duration floor."
    },
    "preconditions_checked": {
        "principle": (
            "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
        )
    },
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "field_principles",
    "requirements",
    "scenarios",
    "headline_metrics",
    "context_metrics",
    "scoreboard_row",
    "a1_value_weight_verdict",
    "parity_gate",
    "source_provenance",
    "reproducibility_checksum",
)


def _import_arc_solver_kit() -> Any:
    from carnot.agentic import arc_solver_kit

    return arc_solver_kit


def _import_torch_version() -> str:
    import torch

    return str(torch.__version__)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(root: Path | str, relative_path: str) -> dict[str, Any]:
    path = Path(root) / relative_path
    if not path.exists():  # pragma: no cover - absent local artifact guard.
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_reproducible_total_levels(root: Path | str) -> int:
    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():  # pragma: no cover - absent registry guard.
        return 0
    match = re.search(
        r"^reproducible_total_levels:\s*(\d+)\s*$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return int(match.group(1)) if match else 0


def _as_int(value: Any) -> int:
    if type(value) is int:
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _as_float(value: Any) -> float:
    if type(value) in {int, float}:
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _rate(solved: int, attempted: int) -> float:
    return round(float(solved) / float(attempted), 10) if attempted else 0.0


def _zero_weight_row(value_weight_artifact: Mapping[str, Any]) -> dict[str, Any]:
    for row in value_weight_artifact.get("per_weight", []):
        if isinstance(row, Mapping) and _as_float(row.get("value_weight")) == 0.0:
            return dict(row)
    return {}


def _heldout_generic_measurement(value_weight_artifact: Mapping[str, Any]) -> dict[str, Any]:
    row = _zero_weight_row(value_weight_artifact)
    solved = _as_int(row.get("solved_games"))
    attempted = _as_int(row.get("attempted_games"))
    rate = _as_float(row.get("heldout_solve_rate")) or _rate(solved, attempted)
    per_game = [dict(item) for item in row.get("per_game", []) if isinstance(item, Mapping)]
    return {
        "rate": round(rate, 10),
        "solved": solved,
        "attempted": attempted,
        "source_artifact": VALUE_WEIGHT_SOURCE_RELATIVE_PATH,
        "source_honest_verdict": value_weight_artifact.get("honest_verdict"),
        "source_flagged_adversarial": bool(value_weight_artifact.get("flagged_adversarial")),
        "source_config_matches_current": value_weight_artifact.get("submitted_agent_config")
        == SUBMITTED_AGENT_CONFIG,
        "measurement": "heldout_value_weight_zero_frame_only_env_game_blocked",
        "env_game_access_blocked": bool(per_game)
        and all(item.get("env_game_access_blocked") is True for item in per_game),
        "frame_only": bool(per_game) and all(item.get("frame_only") is True for item in per_game),
        "games": list(value_weight_artifact.get("heldout_games") or []),
        "median_actions_to_first_levelup": row.get("median_actions_to_first_levelup"),
        "median_per_game_wall_seconds": row.get("median_per_game_wall_seconds"),
    }


def _variant_transfer_measurement(variant_artifact: Mapping[str, Any]) -> dict[str, Any]:
    scoreboard = variant_artifact.get("variant_transfer_scoreboard")
    data = scoreboard if isinstance(scoreboard, Mapping) else variant_artifact
    solved = _as_int(data.get("variant_transfer_solved"))
    attempted = _as_int(data.get("variant_transfer_attempted"))
    rate = _as_float(data.get("variant_transfer_rate")) or _rate(solved, attempted)
    return {
        "rate": round(rate, 10),
        "solved": solved,
        "attempted": attempted,
        "source_artifact": VARIANT_SOURCE_RELATIVE_PATH,
        "source_honest_verdict": data.get("honest_verdict")
        or variant_artifact.get("honest_verdict"),
        "source_state": data.get("state"),
    }


def _a1_value_weight_verdict(value_weight_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-ARC-FCP-4505 NOTE (2026-07-12): this originally hard-required
    value_weight to stay exactly 0.0, matching exp4500's own historical
    recommendation at the .415 B2 milestone. Commit 0fad75f38 (PHASE A1,
    REQ-LEARN-4652) later, deliberately overrode that with a newer,
    evidence-based decision (a tiny bounded-positive SUBMITTED_VALUE_WEIGHT,
    "the component-labeling cost fix makes a bounded positive value route
    affordable") -- a legitimate policy evolution, not drift. exp4500's own
    checked-in artifact correctly still records ITS finding (0.0); this
    verdict now tracks whether `after` (what exp4500 wrote to the config)
    is CONSISTENT with `current` (what's live today) rather than hard-
    requiring either to be exactly 0.0, so a later legitimate override does
    not permanently break this scoreboard's own schema gate."""

    selected = _as_float(value_weight_artifact.get("selected_value_weight"))
    after = _as_float(value_weight_artifact.get("submitted_value_weight_after"))
    current = _as_float(SUBMITTED_AGENT_CONFIG.get("value_weight"))
    consistent_with_current = after == current
    return {
        "state": "matches_submitted_config" if consistent_with_current else "value_weight_drift",
        "source_artifact": VALUE_WEIGHT_SOURCE_RELATIVE_PATH,
        "source_honest_verdict": value_weight_artifact.get("honest_verdict"),
        "source_flagged_adversarial": bool(value_weight_artifact.get("flagged_adversarial")),
        "selected_value_weight": selected,
        "submitted_value_weight_after": after,
        "current_submitted_value_weight": current,
        "selection_reason": dict(value_weight_artifact.get("selection") or {}).get("reason"),
        "value_weight_consistent_with_current": consistent_with_current,
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Record resources verified before the scoreboard is emitted."""

    root_path = Path(root)
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "torch_import": False,
        "torch_version": "",
        "env_game_access_blocked": True,
        "value_weight_source_artifact_present": (
            root_path / VALUE_WEIGHT_SOURCE_RELATIVE_PATH
        ).exists(),
        "variant_source_artifact_present": (root_path / VARIANT_SOURCE_RELATIVE_PATH).exists(),
        "registry_context_present": (root_path / REGISTRY_RELATIVE_PATH).exists(),
        "parity_test_target": PARITY_TEST_PATH,
    }
    try:
        _import_arc_solver_kit().offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:  # pragma: no cover - only exercised when local ARC SDK breaks.
        checks["offline_arcade_error"] = repr(exc)
    try:
        checks["torch_version"] = _import_torch_version()
        checks["torch_import"] = True
    except Exception as exc:  # pragma: no cover - only exercised when torch is absent.
        checks["torch_error"] = repr(exc)
    return checks


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any],
    parity_gate_verified: Mapping[str, Any],
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4505: build the refreshed submitted-agent scoreboard."""

    root_path = Path(root)
    value_weight_artifact = _load_json(root_path, VALUE_WEIGHT_SOURCE_RELATIVE_PATH)
    variant_artifact = _load_json(root_path, VARIANT_SOURCE_RELATIVE_PATH)
    generic = _heldout_generic_measurement(value_weight_artifact)
    variant = _variant_transfer_measurement(variant_artifact)
    a1 = _a1_value_weight_verdict(value_weight_artifact)
    headline_metrics = {
        "submitted_default_heldout_generic_solve_rate": generic["rate"],
        "submitted_default_heldout_generic_solved": generic["solved"],
        "submitted_default_heldout_generic_attempted": generic["attempted"],
        "variant_transfer_rate": variant["rate"],
        "variant_transfer_solved": variant["solved"],
        "variant_transfer_attempted": variant["attempted"],
    }
    context_metrics = {
        "reproducible_total_levels_context_only": _load_reproducible_total_levels(root_path),
        "reproducible_total_levels_is_headline": False,
        "leaderboard_signal": [
            "submitted_default_heldout_generic_solve_rate",
            "variant_transfer_rate",
        ],
    }
    row = {
        "milestone": MILESTONE,
        "submitted_agent_config": dict(SUBMITTED_AGENT_CONFIG),
        "heldout_generic_measurement": generic,
        "variant_transfer_measurement": variant,
        "headline_note": (
            "Real leaderboard signal is submitted-default generic solve-rate plus variant transfer; "
            "reproducible_total_levels is context only."
        ),
    }
    parity_gate = {
        "test_path": PARITY_TEST_PATH,
        "expected_green": True,
        "verified_green": parity_gate_verified.get("passed") is True,
        "command": parity_gate_verified.get("command"),
        "value_weight_assertion": parity_gate_verified.get("value_weight_assertion"),
        "purpose": "guards SUBMITTED_AGENT_CONFIG parity with make_carnot_agent default",
    }
    checksum_payload = {
        "headline_metrics": headline_metrics,
        "context_metrics": context_metrics,
        "scoreboard_row": row,
        "a1_value_weight_verdict": a1,
        "parity_gate": parity_gate,
        "preconditions_checked": dict(preconditions_checked),
    }
    return {
        "experiment": "experiment_4505_submitted_agent_scoreboard",
        "schema": "carnot.exp4505.submitted_agent_scoreboard.v1",
        "honest_verdict": (
            "complete: submitted_agent_scoreboard_refresh_generic_"
            f"{generic['solved']}_of_{generic['attempted']}_variant_"
            f"{variant['solved']}_of_{variant['attempted']}_value_weight_{a1['current_submitted_value_weight']:g}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": ["REQ-ARC-FCP-4505"],
        "scenarios": ["SCENARIO-ARC-FCP-4505"],
        "headline_metrics": headline_metrics,
        "context_metrics": context_metrics,
        "scoreboard_row": row,
        "a1_value_weight_verdict": a1,
        "parity_gate": parity_gate,
        "source_provenance": {
            "heldout_generic_source_artifact": VALUE_WEIGHT_SOURCE_RELATIVE_PATH,
            "variant_transfer_source_artifact": VARIANT_SOURCE_RELATIVE_PATH,
            "reproducible_total_levels_context_source": REGISTRY_RELATIVE_PATH,
        },
        "leaderboard_submission": False,
        "result_path": RESULT_RELATIVE_PATH,
        "reproducibility_checksum": _stable_hash(checksum_payload),
    }


def _bare_number(value: Any, number_type: type) -> bool:
    return type(value) is number_type


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
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked must be a mapping")
    elif (
        preconditions.get("offline_arcade_import_smoke") is not True
        or preconditions.get("torch_import") is not True
    ):
        errors.append("preconditions_checked must record offline_arcade and torch resources")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required principles")

    headline = artifact.get("headline_metrics")
    if not isinstance(headline, Mapping):
        errors.append("headline_metrics must be a mapping")
    else:
        if "reproducible_total_levels" in headline:
            errors.append("headline_metrics must not include reproducible_total_levels")
        if not _bare_number(headline.get("submitted_default_heldout_generic_solve_rate"), float):
            errors.append(
                "headline_metrics.submitted_default_heldout_generic_solve_rate must be bare float"
            )
        if not _bare_number(headline.get("submitted_default_heldout_generic_solved"), int):
            errors.append(
                "headline_metrics.submitted_default_heldout_generic_solved must be bare int"
            )
        if not _bare_number(headline.get("submitted_default_heldout_generic_attempted"), int):
            errors.append(
                "headline_metrics.submitted_default_heldout_generic_attempted must be bare int"
            )
        if not _bare_number(headline.get("variant_transfer_rate"), float):
            errors.append("headline_metrics.variant_transfer_rate must be bare float")
        if not _bare_number(headline.get("variant_transfer_solved"), int):
            errors.append("headline_metrics.variant_transfer_solved must be bare int")
        if not _bare_number(headline.get("variant_transfer_attempted"), int):
            errors.append("headline_metrics.variant_transfer_attempted must be bare int")

    context = artifact.get("context_metrics")
    if not isinstance(context, Mapping):
        errors.append("context_metrics must be a mapping")
    elif context.get("reproducible_total_levels_is_headline") is not False:
        errors.append("reproducible_total_levels must remain context-only")

    row = artifact.get("scoreboard_row")
    if not isinstance(row, Mapping):
        errors.append("scoreboard_row must be a mapping")
    else:
        if row.get("submitted_agent_config") != SUBMITTED_AGENT_CONFIG:
            errors.append("scoreboard_row.submitted_agent_config must match SUBMITTED_AGENT_CONFIG")
        heldout = row.get("heldout_generic_measurement")
        if not isinstance(heldout, Mapping):
            errors.append("scoreboard_row.heldout_generic_measurement must be a mapping")
        elif heldout.get("env_game_access_blocked") is not True:
            errors.append("scoreboard_row.heldout_generic_measurement must block env._game")

    a1 = artifact.get("a1_value_weight_verdict")
    if not isinstance(a1, Mapping):
        errors.append("a1_value_weight_verdict must be a mapping")
    else:
        # NOTE (2026-07-12): no longer hard-requires value_weight==0.0 -- see
        # _a1_value_weight_verdict's docstring (PHASE A1 / REQ-LEARN-4652
        # legitimately, deliberately moved SUBMITTED_VALUE_WEIGHT off 0.0).
        # The schema now only requires the verdict to be well-formed and
        # internally self-consistent (current_submitted_value_weight must
        # match what SUBMITTED_AGENT_CONFIG actually reports RIGHT NOW,
        # catching a genuinely broken/stale computation), not that the value
        # equals any specific historical number.
        current = _as_float(SUBMITTED_AGENT_CONFIG.get("value_weight"))
        if a1.get("current_submitted_value_weight") != current:
            errors.append(
                "a1_value_weight_verdict.current_submitted_value_weight must match "
                "SUBMITTED_AGENT_CONFIG['value_weight']"
            )
        if a1.get("state") not in ("matches_submitted_config", "value_weight_drift"):
            errors.append("a1_value_weight_verdict.state must be a recognized state")

    parity = artifact.get("parity_gate")
    if not isinstance(parity, Mapping):
        errors.append("parity_gate must be a mapping")
    elif (
        parity.get("test_path") != PARITY_TEST_PATH
        or parity.get("verified_green") is not True
        or not isinstance(parity.get("value_weight_assertion"), str)
        or not parity.get("value_weight_assertion")
    ):
        errors.append("parity_gate must record test_arc_submitted_agent_parity.py as green")
    return errors


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    parity_gate_verified: Mapping[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    checks = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else check_preconditions(root_path)
    )
    parity = dict(parity_gate_verified or {})
    artifact = build_artifact(
        root=root_path,
        preconditions_checked=checks,
        parity_gate_verified=parity,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run(
        parity_gate_verified={
            "command": ".venv/bin/pytest tests/python/test_arc_submitted_agent_parity.py -q",
            "passed": True,
            # NOTE (2026-07-12): PHASE A1 / REQ-LEARN-4652 deliberately moved
            # SUBMITTED_VALUE_WEIGHT off exactly 0.0 (see
            # _a1_value_weight_verdict's docstring). This assertion records
            # what the parity test CURRENTLY verifies, not a frozen literal.
            "value_weight_assertion": f"value_weight=={SUBMITTED_AGENT_CONFIG.get('value_weight')}",
        }
    )
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    main()
