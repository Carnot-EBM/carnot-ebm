"""Tests for Exp 2505 paper-v6 capstone (milestone 2026.05.241).

Spec: REQ-REPORT-2505, SCENARIO-REPORT-2505.

These tests cover the synthesis machinery only; the real-input run
against the live ``results/`` tree is what produces the deliverable
artifact and is tested separately via the conductor's research_step
verdict path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import paper_v6_capstone_2505 as exp2505


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    """Helper: write a JSON artifact under tmp_path."""

    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _seed_minimum_corpus(root: Path) -> None:
    """Seed a baseline corpus where Phase 4 fails and AUROC is verified.

    This matches the actual .241 outcome shape: Gate 4 met, Gate 3 unmet.
    """

    _write_json(
        root,
        "results/experiment_2497_phase4_spilled_energy.json",
        {
            "pearson_spilled": -0.022,
            "auroc_spilled": 0.490,
            "phase4_validated_via_spilled": False,
            "tier0q_viable": False,
            "honest_verdict": "complete: phase4_validated=False",
        },
    )
    _write_json(
        root,
        "results/experiment_2498_auroc_adversarial_v2_group_cond.json",
        {
            "group_conditional_auroc_replicated": 0.975,
            "auroc_adversarially_verified": True,
            "honest_verdict": "complete: 0.9750",
        },
    )


def test_capstone_artifact_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-2505: deliverable contains every required artifact field."""

    _seed_minimum_corpus(tmp_path)
    artifact = exp2505.run(
        root=tmp_path,
        out_path=tmp_path / "results" / exp2505.OUTPUT_FILENAME,
    )
    assert exp2505.REQUIRED_ARTIFACT_FIELDS.issubset(set(artifact))
    assert (tmp_path / "results" / exp2505.OUTPUT_FILENAME).is_file()


def test_capstone_honest_verdict_terminal_prefix(tmp_path: Path) -> None:
    """REQ-REPORT-2505: verdict prefix discipline per CLAUDE.md."""

    _seed_minimum_corpus(tmp_path)
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["honest_verdict"].startswith("complete:")


def test_arxiv_not_ready_when_phase4_unvalidated(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2505: arxiv_ready=False when Gate 3 unmet.

    Matches the actual .241 shape: AUROC verified but Phase 4 not.
    """

    _seed_minimum_corpus(tmp_path)
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["auroc_adversarially_verified"] is True
    assert artifact["phase4_validated_any"] is False
    assert artifact["arxiv_ready"] is False
    assert artifact["arxiv_gates"]["gate_3_phase4"] is False
    assert artifact["arxiv_gates"]["gate_4_auroc"] is True


def test_arxiv_ready_when_all_four_gates_met(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2505: arxiv_ready=True iff all 4 gates True.

    Hypothetical: Spilled Energy validates Phase 4 AND exp2498 confirms
    AUROC adversarially. This is the operator-submit case.
    """

    _write_json(
        tmp_path,
        "results/experiment_2497_phase4_spilled_energy.json",
        {
            "phase4_validated_via_spilled": True,
            "tier0q_viable": True,
            "honest_verdict": "complete: phase4_validated=True",
        },
    )
    _write_json(
        tmp_path,
        "results/experiment_2498_auroc_adversarial_v2_group_cond.json",
        {
            "group_conditional_auroc_replicated": 0.975,
            "auroc_adversarially_verified": True,
            "honest_verdict": "complete: 0.9750",
        },
    )
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["phase4_validated_any"] is True
    assert artifact["arxiv_ready"] is True
    assert all(artifact["arxiv_gates"].values())


def test_best_auroc_takes_max_of_replicated_and_ensemble(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2505: best_241_auroc = max(exp2498, exp2499 if present)."""

    _seed_minimum_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2499_spilled_energy_tier0q_ensemble_v6.json",
        {
            "ensemble_v6_auroc_mean": 0.984,
            "tier0q_improves_ensemble": True,
            "honest_verdict": "complete: 0.984",
        },
    )
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["best_241_auroc"] == pytest.approx(0.984, abs=1e-6)
    assert "exp2499" in artifact["best_241_auroc_source"]


def test_missing_artifacts_surface_explicitly(tmp_path: Path) -> None:
    """Capstone records missing inputs rather than fabricating values."""

    # No artifacts seeded — capstone should still produce a valid file.
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    missing_keys = {entry["key"] for entry in artifact["missing_source_artifacts"]}
    assert "phase4_prc_v3" in missing_keys
    assert "phase4_spilled" in missing_keys
    assert artifact["auroc_adversarially_verified"] is False
    assert artifact["phase4_validated_any"] is False
    assert artifact["arxiv_ready"] is False


def test_hardware_status_polarfire_terminal(tmp_path: Path) -> None:
    """PolarFire terminal_state_reached when polarfire_terminal=True."""

    _seed_minimum_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2501_polarfire_terminal.json",
        {"polarfire_terminal": True, "energy_sanity_check_passed": True},
    )
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["polarfire_status"] == "terminal_state_reached"


def test_hardware_status_kv260_pynq_viable(tmp_path: Path) -> None:
    """KV260 pynq_path_viable when exp2502 confirms the bypass."""

    _seed_minimum_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2502_kv260_pynq_sdcard.json",
        {"pynq_path_viable": True},
    )
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["kv260_status"] == "pynq_path_viable"


def test_tier0r_viable_passes_through(tmp_path: Path) -> None:
    """tier0r_viable + tier0r_auroc propagate from exp2504."""

    _seed_minimum_corpus(tmp_path)
    _write_json(
        tmp_path,
        "results/experiment_2504_curry_howard_tier0r.json",
        {"tier0r_viable": True, "tier0r_auroc": 0.9123},
    )
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    assert artifact["tier0r_viable"] is True
    assert artifact["tier0r_auroc"] == pytest.approx(0.9123, abs=1e-6)


def test_validate_artifact_rejects_missing_fields() -> None:
    """validate_artifact crashes loudly on schema violations."""

    with pytest.raises(ValueError, match="missing required fields"):
        exp2505.validate_artifact({"status": "complete"})


def test_validate_artifact_rejects_non_terminal_verdict() -> None:
    """validate_artifact rejects honest_verdict without 'complete:' prefix."""

    payload = {field: None for field in exp2505.REQUIRED_ARTIFACT_FIELDS}
    payload.update(
        {
            "status": "complete",
            "honest_verdict": "partial: did not finish",
            "duration_s": 0.0,
            "phase1_ship_gate_met": True,
            "best_241_auroc": 0.5,
        }
    )
    with pytest.raises(ValueError, match="complete:"):
        exp2505.validate_artifact(payload)


def test_load_artifact_handles_missing_file(tmp_path: Path) -> None:
    """_load_artifact returns None on missing files (no crash)."""

    assert exp2505._load_artifact(tmp_path, "results/nonexistent.json") is None


def test_load_artifact_handles_corrupt_json(tmp_path: Path) -> None:
    """_load_artifact tolerates JSON with trailing garbage."""

    path = tmp_path / "results" / "exp_corrupt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"valid": true}\nGARBAGE', encoding="utf-8")
    result = exp2505._load_artifact(tmp_path, "results/exp_corrupt.json")
    assert result == {"valid": True}


def test_load_artifact_returns_none_for_totally_invalid_json(tmp_path: Path) -> None:
    """_load_artifact returns None for unparseable content."""

    path = tmp_path / "results" / "exp_bad.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("totally not json at all", encoding="utf-8")
    assert exp2505._load_artifact(tmp_path, "results/exp_bad.json") is None


def test_float_field_rejects_bool() -> None:
    """_float_field returns None for bool values (avoids 1.0/0.0 coercion)."""

    assert exp2505._float_field({"x": True}, "x") is None
    assert exp2505._float_field({"x": False}, "x") is None
    assert exp2505._float_field({"x": 1.5}, "x") == 1.5
    assert exp2505._float_field({"x": "str"}, "x") is None
    assert exp2505._float_field(None, "x") is None


def test_bool_field_handles_missing() -> None:
    """_bool_field returns False for missing payload or missing key."""

    assert exp2505._bool_field(None, "x") is False
    assert exp2505._bool_field({}, "x") is False
    assert exp2505._bool_field({"x": True}, "x") is True


def test_phase4_explanation_mentions_both_paths(tmp_path: Path) -> None:
    """phase4_explanation describes both exp2496 and exp2497 outcomes."""

    _seed_minimum_corpus(tmp_path)
    artifact = exp2505.run(root=tmp_path, out_path=tmp_path / "out.json")
    explanation = artifact["phase4_explanation"]
    assert "exp2496" in explanation
    assert "exp2497" in explanation
