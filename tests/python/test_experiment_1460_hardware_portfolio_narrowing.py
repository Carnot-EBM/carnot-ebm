"""Tests for Exp 1460 hardware portfolio narrowing.

Spec traces: REQ-HW-049, SCENARIO-HW-049.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.hardware import hardware_portfolio_narrowing as exp1460


def _write_docs(root: Path, *, architecture: bool = True, wishlist: bool = True) -> None:
    """Create the minimum doc markers needed by the portfolio decision gate."""

    architecture_path = root / "_bmad" / "architecture.md"
    wishlist_path = root / "research-hardware-wishlist.md"
    note_path = root / "docs" / "research-notes" / "hardware_portfolio_narrowing.md"
    architecture_path.parent.mkdir(parents=True, exist_ok=True)
    wishlist_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    architecture_path.write_text(
        "Active hardware tracks (Exp 1460)\nDeferred hardware tracks (Exp 1460)\n"
        if architecture
        else "old architecture\n",
        encoding="utf-8",
    )
    wishlist_path.write_text(
        "Active hardware tracks (Exp 1460)\nDeferred hardware tracks (Exp 1460)\n"
        if wishlist
        else "old wishlist\n",
        encoding="utf-8",
    )
    note_path.write_text(
        "hardware portfolio narrowing\nreopen conditions\n",
        encoding="utf-8",
    )


def test_req_hw_049_spec_anchor_exists() -> None:
    """REQ-HW-049, SCENARIO-HW-049: portfolio narrowing is spec-anchored."""

    spec = (exp1460.PROJECT_ROOT / "openspec/capabilities/fpga/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-HW-049" in spec
    assert "SCENARIO-HW-049" in spec
    assert "experiment_1460_hardware_portfolio_narrowing.json" in spec


def test_req_hw_049_writes_in_progress_marker(tmp_path: Path) -> None:
    """REQ-HW-049: interrupted runs leave an explicit in-progress artifact."""

    output = tmp_path / "results" / "experiment_1460_hardware_portfolio_narrowing.json"

    marker = exp1460.write_in_progress_artifact(output)

    assert marker["status"] == "in_progress"
    assert exp1460.REQUIRED_ARTIFACT_FIELDS <= set(marker)
    assert json.loads(output.read_text(encoding="utf-8")) == marker


def test_scenario_hw_049_builds_three_track_decision(tmp_path: Path) -> None:
    """SCENARIO-HW-049: the decision caps active work and gates claims."""

    _write_docs(tmp_path)

    artifact = exp1460.build_portfolio_decision(project_root=tmp_path, run_date="20260507")

    assert exp1460.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["active_hardware_track_count"] == 3
    assert artifact["active_hardware_track_count"] == len(artifact["active_hardware_tracks"])
    assert [track["track_id"] for track in artifact["active_hardware_tracks"]] == [
        "dual_rtx3090_live_sota_runtime",
        "kv260_discrete_sb_rtl_sim",
        "thrml_tsu_compatibility_sim",
    ]
    assert artifact["architecture_updated"] is True
    assert artifact["hardware_wishlist_updated"] is True
    assert artifact["decision_note_path"] == (
        "docs/research-notes/hardware_portfolio_narrowing.md"
    )
    assert "no KV260 board, Extropic, NPU, or photonic execution claim" in artifact[
        "honest_verdict"
    ]
    for track in artifact["active_hardware_tracks"]:
        assert track["claim_boundary"].startswith("No ")
        assert track["evidence"]
        assert track["immediate_research_value"]
        assert track["readiness"]
    for track in artifact["deferred_hardware_tracks"]:
        assert track["reopen_condition"]


def test_req_hw_049_rejects_missing_reopen_conditions(tmp_path: Path) -> None:
    """REQ-HW-049: every deferred track needs a concrete reopen condition."""

    _write_docs(tmp_path)
    artifact = exp1460.build_portfolio_decision(project_root=tmp_path)
    artifact["deferred_hardware_tracks"][0] = dict(
        artifact["deferred_hardware_tracks"][0],
        reopen_condition="",
    )

    with pytest.raises(ValueError, match="reopen_condition"):
        exp1460.validate_artifact(artifact)


def test_req_hw_049_records_missing_doc_updates(tmp_path: Path) -> None:
    """REQ-HW-049: doc-update booleans are evidence-based, not hard-coded."""

    missing_docs = exp1460.build_portfolio_decision(project_root=tmp_path)
    assert missing_docs["architecture_updated"] is False
    assert missing_docs["hardware_wishlist_updated"] is False

    _write_docs(tmp_path, architecture=False, wishlist=True)

    artifact = exp1460.build_portfolio_decision(project_root=tmp_path)

    assert artifact["architecture_updated"] is False
    assert artifact["hardware_wishlist_updated"] is True
    with pytest.raises(ValueError, match="architecture_updated"):
        exp1460.validate_artifact(artifact)

    _write_docs(tmp_path, architecture=True, wishlist=False)
    wishlist_missing = exp1460.build_portfolio_decision(project_root=tmp_path)
    assert wishlist_missing["architecture_updated"] is True
    assert wishlist_missing["hardware_wishlist_updated"] is False
    with pytest.raises(ValueError, match="hardware_wishlist_updated"):
        exp1460.validate_artifact(wishlist_missing)


def test_scenario_hw_049_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-049: run_experiment writes the complete JSON artifact."""

    _write_docs(tmp_path)
    output = tmp_path / "results" / "experiment_1460_hardware_portfolio_narrowing.json"

    artifact = exp1460.run_experiment(
        project_root=tmp_path,
        output_path=output,
        run_date="20260507",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    exp1460.validate_artifact(artifact)


def test_req_hw_049_validator_rejects_schema_drift(tmp_path: Path) -> None:
    """REQ-HW-049: schema drift cannot silently become a complete artifact."""

    _write_docs(tmp_path)
    artifact = exp1460.build_portfolio_decision(project_root=tmp_path)

    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1460.validate_artifact(missing_required)

    wrong_status = dict(artifact, status="in_progress")
    with pytest.raises(ValueError, match="status"):
        exp1460.validate_artifact(wrong_status)

    active_not_list = dict(artifact, active_hardware_tracks="gpu")
    with pytest.raises(ValueError, match="active_hardware_tracks"):
        exp1460.validate_artifact(active_not_list)

    count_mismatch = dict(artifact, active_hardware_track_count=2)
    with pytest.raises(ValueError, match="active_hardware_track_count"):
        exp1460.validate_artifact(count_mismatch)

    too_many_tracks = dict(
        artifact,
        active_hardware_tracks=[
            *artifact["active_hardware_tracks"],
            dict(artifact["active_hardware_tracks"][0], track_id="extra"),
        ],
        active_hardware_track_count=4,
    )
    with pytest.raises(ValueError, match="between 2 and 3"):
        exp1460.validate_artifact(too_many_tracks)

    no_deferred = dict(artifact, deferred_hardware_tracks=[])
    with pytest.raises(ValueError, match="deferred_hardware_tracks"):
        exp1460.validate_artifact(no_deferred)

    wrong_note = dict(artifact, decision_note_path="docs/wrong.md")
    with pytest.raises(ValueError, match="decision_note_path"):
        exp1460.validate_artifact(wrong_note)

    weak_verdict = dict(artifact, honest_verdict="active_tracks_narrowed")
    with pytest.raises(ValueError, match="claim boundary"):
        exp1460.validate_artifact(weak_verdict)


def test_req_hw_049_validator_rejects_active_track_drift(tmp_path: Path) -> None:
    """REQ-HW-049: active track records need evidence and claim boundaries."""

    _write_docs(tmp_path)
    artifact = exp1460.build_portfolio_decision(project_root=tmp_path)

    missing_label = dict(artifact["active_hardware_tracks"][0])
    missing_label.pop("label")
    active_missing = dict(
        artifact,
        active_hardware_tracks=[missing_label, *artifact["active_hardware_tracks"][1:]],
    )
    with pytest.raises(ValueError, match="missing"):
        exp1460.validate_artifact(active_missing)

    no_evidence = dict(artifact["active_hardware_tracks"][0], evidence=[])
    active_no_evidence = dict(
        artifact,
        active_hardware_tracks=[no_evidence, *artifact["active_hardware_tracks"][1:]],
    )
    with pytest.raises(ValueError, match="evidence"):
        exp1460.validate_artifact(active_no_evidence)

    weak_boundary = dict(
        artifact["active_hardware_tracks"][0],
        claim_boundary="Maybe claim later",
    )
    active_weak_boundary = dict(
        artifact,
        active_hardware_tracks=[weak_boundary, *artifact["active_hardware_tracks"][1:]],
    )
    with pytest.raises(ValueError, match="claim_boundary"):
        exp1460.validate_artifact(active_weak_boundary)
