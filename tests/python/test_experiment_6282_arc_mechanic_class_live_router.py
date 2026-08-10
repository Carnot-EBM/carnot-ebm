"""Tests for Exp6282 mechanic-class live inducer route.

Spec refs: REQ-ARC-WMTE-6282,
SCENARIO-ARC-WMTE-6282-GAME-BLIND-FIXTURES,
SCENARIO-ARC-WMTE-6282-LIVE-PROMPT-ROUTE,
SCENARIO-ARC-WMTE-6282-ARTIFACT-PROVENANCE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6282_arc_mechanic_class_live_router as exp6282
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_mechanic_class_detector as detector


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _push_transitions() -> list[e3.Transition]:
    rows: list[e3.Transition] = []
    for x in (1, 2, 3):
        before = np.zeros((5, 7), dtype=int)
        after = np.zeros((5, 7), dtype=int)
        before[2, x] = 1
        before[2, x + 1] = 2
        after[2, x + 1] = 1
        after[2, x + 2] = 2
        rows.append(e3.Transition(before, 4, None, after, 0, 0))
    return rows


def test_req_arc_wmte_6282_spec_declares_router_contract() -> None:
    """REQ-ARC-WMTE-6282: OpenSpec names the router scenarios and artifact."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6282") :]
    for marker in (
        "SCENARIO-ARC-WMTE-6282-GAME-BLIND-FIXTURES",
        "SCENARIO-ARC-WMTE-6282-LIVE-PROMPT-ROUTE",
        "SCENARIO-ARC-WMTE-6282-ARTIFACT-PROVENANCE",
        exp6282.RESULT_RELATIVE_PATH.as_posix(),
        "solve_provenance` SHALL equal `live_agent_self_discovery`",
        "`inference_substrate` SHALL equal\n+`live_llm_inference`".replace("\n+", "\n"),
    ):
        assert marker in section
    for field in exp6282.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6282_game_blind_fixture_metrics() -> None:
    """SCENARIO-ARC-WMTE-6282-GAME-BLIND-FIXTURES: controls classify without game ids."""

    fixtures = detector.build_synthetic_fixture_manifest(seed=6282, per_family=5)
    assert detector.fixture_family_counts(fixtures) == {
        "navigation": 5,
        "negative": 5,
        "push_block": 5,
        "toggle_move": 5,
    }
    assert all(fixture.game_id is None for fixture in fixtures)

    metrics = detector.evaluate_detector_on_fixtures(fixtures)

    assert metrics["sample_size"] == 20
    assert metrics["overall_accuracy"] >= 0.95
    assert metrics["by_family"]["push_block"]["correct"] == 5
    assert metrics["by_family"]["toggle_move"]["correct"] == 5
    assert metrics["forbidden_inputs"] == {
        "game_id_used": False,
        "hidden_source_used": False,
        "bfs_used": False,
        "adapter_used": False,
    }


def test_req_arc_wmte_6282_detector_guard_edges() -> None:
    """REQ-ARC-WMTE-6282: detector handles malformed and unknown transition shapes."""

    a = np.zeros((2, 2), dtype=int)
    b = np.zeros((3, 2), dtype=int)
    assert detector._diff_points(a, b) == []
    assert detector._is_push_transition(a, b) is False
    assert detector._single_color_move_count(a, b) == 0
    assert detector._is_toggle_move_transition(a, b) is False
    assert detector._data((a,)) is None

    toggle_before = np.zeros((3, 3), dtype=int)
    toggle_after = toggle_before.copy()
    toggle_after[0, 0] = 1
    toggle_after[0, 1] = 2
    toggle_after[1, 0] = 1
    assert detector._is_toggle_move_transition(toggle_before, toggle_after) is True

    unknown_after = np.zeros((3, 3), dtype=int)
    unknown_after[0, 0] = 4
    unknown = detector.classify_transition_history([(toggle_before, 4, None, unknown_after)])
    assert unknown.predicted_class == "unknown"
    assert detector.prompt_block([(toggle_before, 6, {"x": 1, "y": 1}, toggle_before)]) == ""
    with pytest.raises(TypeError, match="transition"):
        detector.transition_features([object()])


def test_scenario_arc_wmte_6282_live_prompt_route_is_default_off(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-6282-LIVE-PROMPT-ROUTE: disabled route is byte-identical."""

    transitions = _push_transitions()
    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "0")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "0")
    monkeypatch.delenv("CARNOT_ARC_MECHANIC_CLASS_ROUTER", raising=False)
    control = e3.induce_prompt("synthetic", transitions, cell=1)

    monkeypatch.setenv("CARNOT_ARC_MECHANIC_CLASS_ROUTER", "0")
    explicit_off = e3.induce_prompt("synthetic", transitions, cell=1)

    assert explicit_off == control
    assert "MECHANIC CLASS ROUTER" not in control


def test_scenario_arc_wmte_6282_live_prompt_route_adds_uncertain_class(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-6282-LIVE-PROMPT-ROUTE: enabled route appends one class block."""

    monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "0")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", "0")
    monkeypatch.setenv("CARNOT_ARC_MECHANIC_CLASS_ROUTER", "1")

    prompt = e3.induce_prompt("synthetic", _push_transitions(), cell=1)

    assert prompt.count("MECHANIC CLASS ROUTER") == 1
    assert "class=push_block" in prompt
    assert "uncertainty=" in prompt
    assert "support=" in prompt


def test_scenario_arc_wmte_6282_artifact_schema_and_provenance(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6282-ARTIFACT-PROVENANCE: required zero-credit fields are guarded."""

    result_path = tmp_path / exp6282.RESULT_RELATIVE_PATH.name
    manifest_path = tmp_path / exp6282.SYNTHETIC_FIXTURE_MANIFEST_RELATIVE_PATH.name

    artifact = exp6282.run(
        date="20260810",
        result_path=result_path,
        manifest_path=manifest_path,
        duration_s=65.0,
        test_exit_codes={exp6282.RUN_COMMAND: 0},
        llm_runner=exp6282.deterministic_test_llm_runner,
        write=True,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert manifest_path.exists()
    assert set(exp6282.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6282.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert set(exp6282.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])

    assert artifact["status"] == "complete"
    assert artifact["target_was_unsolved_at_precheck"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["game_level_solve_claimed"] is False
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is False
    for field in exp6282.FORBIDDEN_COUNT_FIELDS:
        assert type(artifact[field]) is int
        assert artifact[field] == 0
    assert artifact["route_activation_counts"]["treatment"] == 1
    assert artifact["route_activation_counts"]["control"] == 0
    assert artifact["treatment_fire_receipt"]["fired"] is True
    assert artifact["proposal_coverage_by_arm"]["treatment"] > 0.0
    assert artifact["reproducibility_checksum"] == exp6282.payload_checksum(artifact)


def test_req_arc_wmte_6282_helper_edges(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-6282: helper receipts and rates are deterministic."""

    monkeypatch.setenv("CARNOT_ARC_MECHANIC_CLASS_ROUTER", "old")
    with exp6282._temporary_env({"CARNOT_ARC_MECHANIC_CLASS_ROUTER": "new"}):
        assert exp6282.os.environ["CARNOT_ARC_MECHANIC_CLASS_ROUTER"] == "new"
    assert exp6282.os.environ["CARNOT_ARC_MECHANIC_CLASS_ROUTER"] == "old"

    assert exp6282._invalid_action_rate("ACTION0 ACTION2 ACTION8") == pytest.approx(2 / 3)

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(exp6282, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    receipt_path.write_text(json.dumps({"cmd": 0, "pending": None}), encoding="utf-8")
    assert exp6282._read_external_test_receipts() == {"cmd": 0, "pending": None}
    receipt_path.write_text("{bad", encoding="utf-8")
    assert exp6282._read_external_test_receipts()[exp6282.RUN_COMMAND] is None


def test_req_arc_wmte_6282_validation_rejects_credit_or_counter_drift(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6282: validation refuses solve credit and forbidden access drift."""

    artifact = exp6282.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        manifest_path=tmp_path / "manifest.json",
        duration_s=65.0,
        llm_runner=exp6282.deterministic_test_llm_runner,
        write=False,
    )

    bad_claim = dict(artifact)
    bad_claim["game_level_solve_claimed"] = True
    bad_claim["reproducibility_checksum"] = exp6282.payload_checksum(bad_claim)
    with pytest.raises(ValueError, match="game_level_solve_claimed"):
        exp6282.validate_artifact(bad_claim)

    bad_count = dict(artifact)
    bad_count["exhaustive_bfs_count"] = 1
    bad_count["reproducibility_checksum"] = exp6282.payload_checksum(bad_count)
    with pytest.raises(ValueError, match="exhaustive_bfs_count"):
        exp6282.validate_artifact(bad_count)


def test_req_arc_wmte_6282_validation_fail_closed_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6282: each non-credit artifact guard names its field."""

    artifact = exp6282.run(
        date="20260810",
        result_path=tmp_path / "artifact.json",
        manifest_path=tmp_path / "manifest.json",
        duration_s=65.0,
        llm_runner=exp6282.deterministic_test_llm_runner,
        write=False,
    )

    cases = []
    missing = dict(artifact)
    missing.pop("status")
    cases.append((missing, "missing fields"))

    for field, value, message in (
        ("field_principles", {}, "field_principles"),
        ("field_provenance", {}, "field_provenance"),
        ("solve_provenance", "development_proxy", "solve_provenance"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("target_was_unsolved_at_precheck", False, "target_was_unsolved_at_precheck"),
    ):
        bad = dict(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = exp6282.payload_checksum(bad)
        cases.append((bad, message))

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    cases.append((bad_checksum, "reproducibility_checksum"))

    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            exp6282.validate_artifact(payload)
