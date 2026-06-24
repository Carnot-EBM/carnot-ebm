"""Tests for Exp 4670 multi-level harness CI-gate.

Spec refs: REQ-ARC-WMTE-4670,
SCENARIO-ARC-WMTE-4670-DEGENERATE-METRIC-GATE,
SCENARIO-ARC-WMTE-4670-PORT-HYGIENE,
SCENARIO-ARC-WMTE-4670-FIRST-WIN-FLOOR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


DEGENERATE_ROLLOUT_SOURCE = """
MULTI_LEVEL_TARGET_LEVELS = 1

def run_variant_attempt():
    for _index in range(3):
        if start_level is not None and reached > start_level:
            actions_to_first = actions
            break
"""


FIXED_ROLLOUT_SOURCE = """
MULTI_LEVEL_TARGET_LEVELS = 2

def run_variant_attempt():
    for _index in range(3):
        if start_level is not None and reached > start_level and actions_to_first is None:
            actions_to_first = actions
        if policy.is_done(frames, latest):
            break
"""


def _qwen_artifact(*, served: str, verified: bool = True, port: int = 8920) -> dict[str, Any]:
    return {
        "claimed_model": "Qwen",
        "proposer_served_model": served,
        "metric_harness_fixed": {
            "qwen_port_props_verified": verified,
            "port": port,
        },
    }


def _passed_guard() -> dict[str, Any]:
    return {"passed": True, "errors": []}


def test_req_arc_wmte_4670_spec_declares_multilevel_cigate_contract() -> None:
    """REQ-ARC-WMTE-4670: OpenSpec anchors all guard fields and principles."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4670" in spec
    assert "SCENARIO-ARC-WMTE-4670-DEGENERATE-METRIC-GATE" in spec
    assert "SCENARIO-ARC-WMTE-4670-PORT-HYGIENE" in spec
    assert "SCENARIO-ARC-WMTE-4670-FIRST-WIN-FLOOR" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4670_degenerate_metric_gate_flags_target_and_break() -> None:
    """SCENARIO-ARC-WMTE-4670-DEGENERATE-METRIC-GATE: degenerate rollout fixtures fail."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    explicit = mod.validate_multilevel_rollout_config(
        {"target_levels": 1, "break_at_first_win": True}
    )
    parsed = mod.rollout_config_from_source(DEGENERATE_ROLLOUT_SOURCE)
    fixed = mod.validate_multilevel_rollout_config(
        mod.rollout_config_from_source(FIXED_ROLLOUT_SOURCE)
    )

    assert explicit["passed"] is False
    assert parsed["passed"] is False
    assert "target_levels<2" in explicit["errors"]
    assert "break_at_first_win" in explicit["errors"]
    assert "target_levels<2" in parsed["errors"]
    assert "break_at_first_win" in parsed["errors"]
    with pytest.raises(mod.GateFailure, match="target_levels<2"):
        mod.assert_multilevel_rollout_guard(parsed)
    assert fixed["passed"] is True
    assert fixed["target_levels"] == 2
    assert fixed["break_at_first_win"] is False
    assert mod.assert_multilevel_rollout_guard({"target_levels": 2})["passed"] is True

    annotated = mod.rollout_config_from_source(
        "MULTI_LEVEL_TARGET_LEVELS: int = 2\n"
        "def run_variant_attempt():\n"
        "    return None\n"
    )
    missing_runner = mod.rollout_config_from_source("MULTI_LEVEL_TARGET_LEVELS = 2\n")
    missing_target = mod.rollout_config_from_source("def run_variant_attempt():\n    return None\n")

    assert annotated["passed"] is True
    assert missing_runner["errors"] == ["run_variant_attempt_missing"]
    assert "target_levels<2" in missing_target["errors"]


def test_scenario_arc_wmte_4670_actual_4628_rollout_is_not_degenerate() -> None:
    """REQ-ARC-WMTE-4670: the live 4628 rollout target and source shape stay fixed."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    inspected = mod.inspect_exp4628_rollout(REPO)

    assert inspected["passed"] is True
    assert inspected["target_levels"] >= 2
    assert inspected["break_at_first_win"] is False


def test_scenario_arc_wmte_4670_port_hygiene_flags_gemma_on_qwen_claim() -> None:
    """SCENARIO-ARC-WMTE-4670-PORT-HYGIENE: wrong served model is a hard failure."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    bad = mod.validate_proposer_port_hygiene(
        _qwen_artifact(served="gemma-4-12B-it", port=8919)
    )
    missing_props = mod.validate_proposer_port_hygiene(
        _qwen_artifact(served="Qwen3.5-9B-MTP", verified=False)
    )
    good = mod.validate_proposer_port_hygiene(_qwen_artifact(served="Qwen3.5-9B-MTP"))

    assert bad["passed"] is False
    assert "proposer_served_model_mismatch" in bad["errors"]
    assert "qwen_port_props_verified" in missing_props["errors"]
    with pytest.raises(mod.GateFailure, match="proposer_served_model_mismatch"):
        mod.assert_proposer_port_hygiene(bad)
    assert good["passed"] is True
    assert good["claimed_model"] == "Qwen"
    assert good["proposer_served_model"] == "Qwen3.5-9B-MTP"

    missing = mod.validate_proposer_port_hygiene({})
    gemma = mod.validate_proposer_port_hygiene(
        {"claimed_model": "gemma", "proposer_served_model": "gemma"}
    )

    assert {"claimed_model_missing", "proposer_served_model_missing"}.issubset(
        set(missing["errors"])
    )
    assert gemma["passed"] is True


def test_scenario_arc_wmte_4670_performance_floor_flags_first_win_and_multilevel_drops() -> None:
    """SCENARIO-ARC-WMTE-4670-FIRST-WIN-FLOOR: below-floor metrics fail by name."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    floors = {"first_win_rate": 0.5, "live_multi_level_solve_rate": 0.25}
    regressed = mod.validate_performance_floor(
        {"first_win_rate": 0.49, "live_multi_level_solve_rate": 0.2},
        floors=floors,
    )
    honest = mod.validate_performance_floor(
        {"first_win_rate": 0.5, "live_multi_level_solve_rate": 0.25},
        floors=floors,
    )

    assert regressed["passed"] is False
    assert "first_win_rate_below_floor" in regressed["errors"]
    assert "live_multi_level_solve_rate_below_floor" in regressed["errors"]
    with pytest.raises(mod.GateFailure, match="first_win_rate_below_floor"):
        mod.assert_performance_floor(regressed)
    assert honest["passed"] is True


def test_req_arc_wmte_4670_metric_extraction_reuses_4646_and_a1_level_maps() -> None:
    """REQ-ARC-WMTE-4670: floor metrics come from 4646-compatible rows or A1 levels."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    from_attempts = mod.extract_performance_metrics(
        {
            "coheadline_block": {"first_win_rate": 0.75},
            "live_measurement": {
                "attempts": [
                    {"attempted": True, "depth_of_live_solve": 2},
                    {"attempted": True, "depth_of_live_solve": 1},
                ]
            },
        }
    )
    from_a1 = mod.extract_performance_metrics(
        {"generic_agent_reached_level": {"lp85": 1, "sc25": 0}}
    )

    assert from_attempts["first_win_rate"] == pytest.approx(0.75)
    assert from_attempts["live_multi_level_solve_rate"] == pytest.approx(0.5)
    assert from_a1["first_win_rate"] == pytest.approx(0.5)
    assert from_a1["live_multi_level_solve_rate"] == pytest.approx(0.0)
    assert from_a1["sample_size"] == 2


def test_req_arc_wmte_4670_artifact_schema_and_run_write_terminal_json(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4670: terminal artifact is checksummed and schema-validated."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        degenerate_metric_cigate_added={"passed": True, "errors": []},
        port_hygiene_guard_added={"passed": True, "errors": []},
        first_win_floor_cigate_added={"passed": True, "errors": []},
        tests_added={"passed": True, "test_file": __file__},
        duration_s=1.0,
    )
    written = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": True, "offline_arcade": True},
        source_text=FIXED_ROLLOUT_SOURCE,
        a1_artifact={
            "claimed_model": "Qwen",
            "proposer_served_model": "Qwen3.5-9B-MTP",
            "metric_harness_fixed": {"qwen_port_props_verified": True, "port": 8920},
            "generic_agent_reached_level": {"lp85": 1, "sc25": 0},
        },
        duration_s=1.0,
        write=True,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    blocked = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.0,
        write=False,
    )
    blocked_written = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.0,
        write=True,
    )

    assert artifact["honest_verdict"] == (
        "success: multilevel_harness_cigate_plus_port_hygiene_shipped_tests_green"
    )
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert loaded == written
    assert written["degenerate_metric_cigate_added"]["passed"] is True
    assert written["port_hygiene_guard_added"]["passed"] is True
    assert written["first_win_floor_cigate_added"]["passed"] is True
    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked_written["honest_verdict"] == "blocked_offline_arcade"

    wrong = {**artifact, "verifier_is_oracle": True, "reproducibility_checksum": "bad"}
    wrong["degenerate_metric_cigate_added"] = {"passed": False}
    bad_prefix = {**artifact, "honest_verdict": "done"}
    bad_substrate = {**artifact, "inference_substrate": "aggregation_from_upstream_artifacts"}
    no_principles = {**artifact, "field_principles": None}
    missing_principle = {**artifact, "field_principles": dict(artifact["field_principles"])}
    del missing_principle["field_principles"]["honest_verdict"]

    assert {
        "verifier_is_oracle_false",
        "degenerate_metric_cigate_added",
        "reproducibility_checksum",
    }.issubset(set(mod.artifact_schema_errors(wrong)))
    assert "honest_verdict_terminal_prefix" in mod.artifact_schema_errors(bad_prefix)
    assert "inference_substrate" in mod.artifact_schema_errors(bad_substrate)
    assert "field_principles" in mod.artifact_schema_errors(no_principles)
    assert "field_principles.honest_verdict" in mod.artifact_schema_errors(missing_principle)


def test_req_arc_wmte_4670_run_raises_when_terminal_schema_is_invalid(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """REQ-ARC-WMTE-4670: schema errors make the CI-gate fail loudly."""

    from carnot import experiment_4670_multilevel_harness_cigate as mod

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda _artifact: ["forced_schema_error"])

    with pytest.raises(mod.GateFailure, match="forced_schema_error"):
        mod.run(
            root=tmp_path,
            preconditions_checked={"ok": True, "offline_arcade": True},
            source_text=FIXED_ROLLOUT_SOURCE,
            a1_artifact={
                "claimed_model": "Qwen",
                "proposer_served_model": "Qwen3.5-9B-MTP",
                "metric_harness_fixed": {"qwen_port_props_verified": True, "port": 8920},
                "generic_agent_reached_level": {"lp85": 1, "sc25": 0},
            },
            duration_s=1.0,
            write=False,
        )
