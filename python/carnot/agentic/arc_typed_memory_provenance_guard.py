"""Typed-memory provenance guard for ARC live routing.

Spec refs: REQ-REPORT-5240, SCENARIO-REPORT-5240-LIVE-PATCH-SYNTHESIS,
SCENARIO-REPORT-5240-NO-PATCH-WITHOUT-EVIDENCE.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP5228_RUBRIC_GATE = "results/experiment_5228_arc_provenance_skill_rubric_gate_v478.json"
EXP5239_MEMORY_ABLATION = (
    "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json"
)
TYPED_MEMORY_PROVENANCE_GUARD_HEADS = ("provenance", "failures", "skills_rubrics")


def typed_memory_provenance_guard(root: Path | str = REPO) -> dict[str, Any]:
    """REQ-REPORT-5240: expose the controlled typed-memory guard to live routing."""

    root_path = Path(root)
    rubric_gate = _load_json(root_path, EXP5228_RUBRIC_GATE)
    memory_ablation = _load_json(root_path, EXP5239_MEMORY_ABLATION)
    registry_precheck_done = (root_path / "ops" / "arc_solve_registry.yaml").exists()
    missing = [
        relative
        for relative, payload in (
            (EXP5228_RUBRIC_GATE, rubric_gate),
            (EXP5239_MEMORY_ABLATION, memory_ablation),
        )
        if payload is None
    ]
    if rubric_gate is None or memory_ablation is None:
        return _disabled(
            registry_precheck_done=registry_precheck_done,
            reason=f"missing_or_unreadable_artifacts:{','.join(missing)}",
        )

    recommended_heads = list(_wrapped_value(memory_ablation, "recommended_arc_memory_heads", []))
    controlled_reuse = {
        "aligned_vs_no_memory_delta": float(
            _wrapped_value(memory_ablation, "aligned_vs_no_memory_delta", 0.0) or 0.0
        ),
        "aligned_vs_shuffled_delta": float(
            _wrapped_value(memory_ablation, "aligned_vs_shuffled_delta", 0.0) or 0.0
        ),
        "retention_passed": bool(
            _wrapped_value(memory_ablation, "retention_check_passed", False)
            or (memory_ablation.get("retention") or {}).get("typed_memory_retention_passed")
            or (memory_ablation.get("retention") or {}).get("passed")
        ),
        "rollback_policy_exercised": bool(
            _wrapped_value(memory_ablation, "rollback_policy_exercised", False)
        ),
        "degradation_detected": bool(_wrapped_value(memory_ablation, "degradation_detected", False)),
    }
    blocked_actions = _arc_blocking_actions(memory_ablation)
    required_heads_present = all(
        head in recommended_heads for head in TYPED_MEMORY_PROVENANCE_GUARD_HEADS
    )
    enabled = bool(
        rubric_gate.get("arc_skill_rubric_usable") is True
        and registry_precheck_done
        and required_heads_present
        and controlled_reuse["aligned_vs_no_memory_delta"] > 0.0
        and controlled_reuse["aligned_vs_shuffled_delta"] > 0.0
        and controlled_reuse["retention_passed"]
        and controlled_reuse["rollback_policy_exercised"]
        and controlled_reuse["degradation_detected"]
        and blocked_actions
    )
    if not enabled:
        guard = _disabled(
            registry_precheck_done=registry_precheck_done,
            reason="controlled typed-memory evidence did not clear the provenance-routing guard",
        )
        guard["controlled_reuse"] = controlled_reuse
        return guard
    return {
        "enabled": True,
        "failure_mode_targeted": "provenance_routing",
        "recommended_heads": recommended_heads,
        "blocked_arc_consumer_actions": blocked_actions,
        "registry_precheck_done": registry_precheck_done,
        "controlled_reuse": controlled_reuse,
        "routing_rule": (
            "Before registry promotion or reusable candidate-pool routing, require live-agent "
            "self-discovery provenance and honor rolled-back ARC memory actions as block or "
            "quarantine guardrails."
        ),
        "source_artifacts": _source_artifacts(),
    }


def _disabled(*, registry_precheck_done: bool, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "failure_mode_targeted": "none",
        "recommended_heads": [],
        "blocked_arc_consumer_actions": [],
        "registry_precheck_done": registry_precheck_done,
        "source_artifacts": _source_artifacts()[:2],
        "reason": reason,
    }


def _wrapped_value(payload: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = payload.get(key, default)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value", default)
    return value


def _load_json(root: Path, relative_path: str) -> dict[str, Any] | None:
    path = root / relative_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _arc_blocking_actions(memory_ablation: Mapping[str, Any]) -> list[str]:
    metrics = memory_ablation.get("arm_metrics") if isinstance(memory_ablation, Mapping) else {}
    aligned = metrics.get("aligned_memory") if isinstance(metrics, Mapping) else {}
    rows = aligned.get("rows") if isinstance(aligned, Mapping) else []
    actions: list[str] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or row.get("arc_patch_relevant") is not True:
            continue
        action = str(row.get("selected_action") or row.get("expected_action") or "")
        if action.startswith(("block_", "quarantine_", "retire_", "no_promote_")):
            actions.append(action)
    return sorted(dict.fromkeys(actions))


def _source_artifacts() -> list[str]:
    return [EXP5228_RUBRIC_GATE, EXP5239_MEMORY_ABLATION, "ops/arc_solve_registry.yaml"]
