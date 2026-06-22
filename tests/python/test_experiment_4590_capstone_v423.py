"""Tests for Exp 4590 .423 capstone aggregation.

Spec refs: REQ-CAPSTONE-4590, SCENARIO-CAPSTONE-4590,
SCENARIO-CAPSTONE-4590-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4590_capstone_v423 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1() -> dict[str, Any]:
    return {
        "honest_verdict": "success: live_submittable_count_53_above_33",
        "flagged_adversarial": False,
        "live_submittable_count_baseline": 33,
        "live_submittable_level_count": 53,
        "count_delta": 20,
        "env_adaptive_resolve_recovered": ["sc25"],
        "verifier_is_oracle": False,
        "ready_for_operator_submit": True,
        "refreshed_package_path": "results/experiment_4580_submission_package_live_gap_close.json",
        "offline_reproduced": {"ar25": 1, "sc25": 52},
    }


def _a2() -> dict[str, Any]:
    return {
        "honest_verdict": "success: ar25_L2_offline_reproduced",
        "flagged_adversarial": False,
        "offline_reproduced": True,
        "registry_update": {
            "prior_total_declared": 53,
            "new_total_declared": 54,
            "reconciled_total_delta": 1,
            "target_game": "ar25",
            "updated": True,
        },
    }


def _flagged_a3() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: feature_router_no_value_honest_null_transfer_gap_sharpened",
        "flagged_adversarial": True,
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_router": 0.04,
        "generic_transfer_rate_baseline": 0.04,
        "transfer_delta": 0.0,
        "transfer_ci": [0.0, 0.0],
        "random_route_control_passed": False,
        "false_negative_risk_checked": False,
        "null_delta_methodology_note": "transfer_delta==0.0 is an honest no-value null.",
    }


def _flagged_a4() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: diversity_floor_no_transfer_honest_null_gap_sharpened",
        "flagged_adversarial": True,
        "verifier_is_oracle": False,
        "firstwin_delta": 0,
        "diversity_off_control_passed": True,
        "false_negative_risk_checked": True,
        "null_delta_methodology_note": "firstwin_delta==0 is an honest no-transfer null.",
    }


def _a5() -> dict[str, Any]:
    return {
        "honest_verdict": "success: primitive_persisted_transfer_s5i5_value_added",
        "flagged_adversarial": False,
        "primitive_persisted": {
            "operator": "env_adaptive_resolve_operator",
            "registry_general_gotcha_id": "primitive_env_adaptive_resolve_operator",
        },
        "transfer_games": ["s5i5", "ft09", "sb26"],
        "transfer_value_per_game": {
            "s5i5": {"value_added": True, "drift_recovered": True},
            "ft09": {"value_added": True, "drift_recovered": True},
            "sb26": {"value_added": True, "drift_recovered": True},
        },
        "offline_reproduced": {"new_levels_banked": 0},
        "new_levels_banked": 0,
    }


def _flagged_a6() -> dict[str, Any]:
    return {
        "honest_verdict": "success: integrated_live_submittable_54_above_33",
        "flagged_adversarial": True,
        "live_submittable_level_count_integrated": 54,
        "generic_transfer_rate_integrated": 0.04,
        "generic_transfer_ci_integrated": [0.0, 0.12],
        "ready_for_operator_submit": True,
    }


def _b1() -> dict[str, Any]:
    return {
        "honest_verdict": "shipped: live_submittable_coheadline_wired",
        "flagged_adversarial": False,
        "live_submittable_level_count": 54,
        "reproducible_total_levels": 54,
        "reproducible_vs_submittable_gap": 0,
        "generic_transfer_rate_over_variants": 0.04,
        "generic_transfer_ci": [0.0, 0.1],
        "action_efficiency_score": 1.0,
        "action_efficiency_ci": [1.0, 1.0],
        "refreshed_package_path": "results/experiment_4585_submission_package_integration_gate.json",
    }


def _baseline() -> dict[str, Any]:
    return {"live_total_levels": 33, "games": 18}


def _artifacts() -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1(),
        "A2": _a2(),
        "A3": _flagged_a3(),
        "A4": _flagged_a4(),
        "A5": _a5(),
        "A6": _flagged_a6(),
        "B1": _b1(),
        "LIVE_BASELINE": _baseline(),
    }


def _live_flags() -> dict[str, list[dict[str, Any]]]:
    false_negative = {
        "kind": "FALSE_NEGATIVE_RISK",
        "severity": "warn",
        "detail": "false_negative_risk_open: positive_control_passed=None",
    }
    return {
        "A3": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "generic_transfer_rate_with_router=0.04 and random_route_transfer_rate=0.04",
            },
            false_negative,
        ],
        "A4": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "solve_rate_with_diversity=0.04 and solve_rate_without_diversity=0.04",
            },
            false_negative,
        ],
        "A6": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "generic_transfer_rate_integrated=0.04 and held_out_solve_rate=0.04",
            }
        ],
    }


def test_req_capstone_4590_spec_declares_capstone_contract() -> None:
    """REQ-CAPSTONE-4590: OpenSpec declares the .423 capstone contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4590",
        "SCENARIO-CAPSTONE-4590",
        "SCENARIO-CAPSTONE-4590-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4590_skips_flagged_and_keeps_false_negative_risk_open() -> None:
    """REQ-CAPSTONE-4590: flagged A3/A4/A6 do not contribute headline metrics."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        preconditions_checked={"ok": True, "fixture": True},
        duration_s=0.001,
    )

    assert artifact["scorecard"]["A3"]["included_in_headline"] is False
    assert artifact["scorecard"]["A4"]["included_in_headline"] is False
    assert artifact["scorecard"]["A6"]["included_in_headline"] is False
    assert artifact["generic_transfer_moved"]["moved"] is False
    assert artifact["generic_transfer_moved"]["reason"] == "a3_flagged_false_negative_risk_open"
    assert artifact["flagged_artifacts_handled"]["excluded_artifacts"] == [
        "results/experiment_4582_feature_router_transfer.json",
        "results/experiment_4583_diversity_floor_transfer.json",
        "results/experiment_4585_integration_gate.json",
    ]
    assert set(artifact["flagged_artifacts_handled"]["positive_control_failed_guard"]) == {
        "A3",
        "A4",
    }


def test_scenario_capstone_4590_builds_honest_coheadline_scorecard() -> None:
    """SCENARIO-CAPSTONE-4590: clean A1/B1 evidence moves live score above 33."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        preconditions_checked={"ok": True, "fixture": True},
        duration_s=0.001,
    )

    assert artifact["honest_verdict"].startswith("success: live_submittable_above_33")
    assert artifact["live_submittable_moved"]["moved"] is True
    assert artifact["live_submittable_moved"]["baseline"] == 33
    assert artifact["live_submittable_moved"]["a1_live_submittable_level_count"] == 53
    assert artifact["live_submittable_moved"]["a1_count_delta"] == 20
    assert artifact["live_submittable_level_count"] == 54
    assert artifact["reproducible_total_levels"] == 54
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.04)
    assert artifact["generic_transfer_ci"] == [0.0, 0.1]
    assert artifact["action_efficiency_score"] == pytest.approx(1.0)
    assert artifact["action_efficiency_ci"] == [1.0, 1.0]
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["scorecard"]["A5"]["new_levels_banked"] == 0
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4590_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4590: run writes the requested capstone JSON from files."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec_path = tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    registry_path = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 54\n", encoding="utf-8")

    artifacts = _artifacts()
    for name, source in mod.UPSTREAM_SOURCES.items():
        _write_json(tmp_path / source.relative_path, artifacts[name])

    artifact = mod.run(
        tmp_path,
        live_flags_by_name=_live_flags(),
        write=True,
        duration_s=0.001,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["live_submittable_level_count"] == 54
    assert written["leaderboard_submission"] is False
    assert mod.validate_artifact(written) == []


def test_req_capstone_4590_schema_errors_fail_closed() -> None:
    """REQ-CAPSTONE-4590: malformed aggregates fail validation instead of celebrating."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        preconditions_checked={"ok": True, "fixture": True},
        duration_s=0.001,
    )

    broken = dict(artifact)
    broken["ready_for_operator_submit"] = "yes"
    broken["reproducible_total_levels_delta"] = "1"

    errors = mod.validate_artifact(broken)
    assert "ready_for_operator_submit must be bare bool" in errors
    assert "reproducible_total_levels_delta must be bare int" in errors

    very_broken = dict(artifact)
    very_broken.update(
        {
            "honest_verdict": "maybe",
            "inference_substrate": "wrong",
            "live_submittable_moved": {"moved": "yes"},
            "generic_transfer_moved": [],
            "generic_transfer_rate_over_variants": -1.0,
            "generic_transfer_ci": [0.0],
            "action_efficiency_score": 2.0,
            "action_efficiency_ci": [1.0],
            "leaderboard_submission": True,
            "verifier_is_oracle_distinct_levers": [],
            "field_principles": {},
            "coheadline_metrics": {**artifact["coheadline_metrics"], "action_efficiency_score": 0.0},
            "reproducibility_checksum": "not-sha",
        }
    )
    richer_errors = mod.validate_artifact(very_broken)
    assert "honest_verdict must be terminal-prefixed" in richer_errors
    assert "inference_substrate mismatch" in richer_errors
    assert "live_submittable_moved.moved must be bare bool" in richer_errors
    assert "generic_transfer_moved must be object" in richer_errors
    assert "generic_transfer_rate_over_variants must be bare float in [0,1]" in richer_errors
    assert "generic_transfer_ci must be [float, float]" in richer_errors
    assert "action_efficiency_score must be bare float in [0,1]" in richer_errors
    assert "action_efficiency_ci must be [float, float]" in richer_errors
    assert "leaderboard_submission must be false" in richer_errors
    assert "verifier_is_oracle_distinct_levers must be object" in richer_errors
    assert "missing field principle for honest_verdict" in richer_errors
    assert "coheadline_metrics.action_efficiency_score must match top-level field" in richer_errors
    assert "reproducibility_checksum must be sha256-prefixed" in richer_errors

    alternate_broken = dict(artifact)
    alternate_broken.update(
        {
            "live_submittable_moved": [],
            "generic_transfer_moved": {"moved": "yes"},
            "field_principles": [],
        }
    )
    alternate_errors = mod.validate_artifact(alternate_broken)
    assert "live_submittable_moved must be object" in alternate_errors
    assert "generic_transfer_moved.moved must be bare bool" in alternate_errors
    assert "field_principles missing" in alternate_errors

    with pytest.raises(ValueError, match="ready_for_operator_submit must be bare bool"):
        mod.write_artifact(artifact=broken)


def test_req_capstone_4590_defensive_helpers_and_alternate_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4590: helper branches stay explicit and deterministic."""

    assert mod._read_json(tmp_path / "missing.json") == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(list_json) == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}

    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[", encoding="utf-8")
    assert mod._read_yaml(bad_yaml) == {}
    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("1\n", encoding="utf-8")
    assert mod._read_yaml(scalar_yaml) == {}
    assert mod._file_sha256(tmp_path / "missing.bin") is None
    assert mod._payload_sha256({"x": 1}).startswith("sha256:")
    assert mod._as_int(True, 7) == 7
    assert mod._as_int("x", 7) == 7
    assert mod._as_float(False, 0.5) == pytest.approx(0.5)
    assert mod._as_float("x", 0.5) == pytest.approx(0.5)
    assert mod._as_float_ci(["x"], default=(1.0, 2.0)) == [1.0, 2.0]

    class BadReader:
        @staticmethod
        def _live_flags(path: Path) -> list[dict[str, Any]]:
            raise RuntimeError(path)

        @staticmethod
        def classify_known_false_positive_null_delta(
            artifact: Mapping[str, Any],
            flags: list[dict[str, Any]],
        ) -> None:
            raise RuntimeError((artifact, flags))

    monkeypatch.setattr(mod, "artifact_reader", None)
    assert mod._live_flags(tmp_path / "x.json") == []
    assert mod._null_delta_corrigendum({}, []) is None
    monkeypatch.setattr(mod, "artifact_reader", BadReader)
    assert mod._live_flags(tmp_path / "x.json") == []
    assert mod._null_delta_corrigendum({}, []) is None

    class CorrigendumReader:
        @staticmethod
        def classify_known_false_positive_null_delta(
            artifact: Mapping[str, Any],
            flags: list[dict[str, Any]],
        ) -> dict[str, Any]:
            return {"kind": "KNOWN_FALSE_POSITIVE_NULL_DELTA_TAUTOLOGY"}

    monkeypatch.setattr(mod, "artifact_reader", CorrigendumReader)
    assert mod._acceptance_gate_failures({"acceptance_gate_main": False}) == [
        "acceptance_gate_main"
    ]
    source = mod.SourceSpec("X", "results/x.json", "fixture")
    missing_status = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={},
        exists=False,
        live_flags_by_name={},
    )
    assert missing_status["reason"] == "missing"
    gate_status = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={"gate_main": False},
        exists=True,
        live_flags_by_name={},
    )
    assert gate_status["reason"] == "failed_acceptance_gate"
    diag_status = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={"flagged_adversarial": True},
        exists=True,
        live_flags_by_name={
            "X": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "control==best null-delta"}]
        },
    )
    assert diag_status["reason"] == "flagged_null_delta_corrigendum_diagnosis_only"
    handled = mod._flagged_artifacts_handled(
        {"X": diag_status, "Y": gate_status, "LIVE_BASELINE": missing_status}
    )
    assert handled["null_delta_carveouts"]
    assert handled["failed_acceptance_gate_overrides"] == [
        {"name": "Y", "artifact": "results/x.json", "failed_gates": ["gate_main"]}
    ]

    assert mod._a2_repro_delta({}, False) == (0, 0, 0)
    assert mod._a2_repro_delta({}, True) == (0, 0, 0)
    assert mod._a2_repro_delta(
        {
            "offline_reproduced": False,
            "registry_update": {
                "prior_total_declared": 1,
                "new_total_declared": 3,
                "updated": False,
            },
        },
        True,
    ) == (1, 3, 0)
    assert mod._a2_repro_delta(
        {
            "offline_reproduced": True,
            "registry_update": {
                "prior_total_declared": 1,
                "new_total_declared": 3,
                "updated": True,
            },
        },
        True,
    ) == (1, 3, 2)
    assert mod._a4_repro_delta({"offline_reproduced": True, "firstwin_delta": 2}, True) == 2
    assert mod._a4_repro_delta({"offline_reproduced": {"new_levels_banked": 1}}, True) == 1
    assert mod._a4_repro_delta({}, True) == 0

    moved = mod._generic_transfer_moved(
        {
            "generic_transfer_rate_with_router": 0.08,
            "transfer_delta": 0.04,
            "transfer_ci": [0.01, 0.07],
            "random_route_control_passed": True,
        },
        {"included_in_headline": True},
    )
    assert moved["moved"] is True

    fallback_artifacts = _artifacts()
    fallback_artifacts["B1"] = {**_b1(), "refreshed_package_path": ""}
    fallback = mod.build_artifact(
        artifacts=fallback_artifacts,
        live_flags_by_name=_live_flags(),
        preconditions_checked={"ok": True},
        duration_s=0.001,
    )
    assert fallback["ready_for_operator_submit"] is True

    null_artifacts = _artifacts()
    null_artifacts["A1"] = {
        **_a1(),
        "live_submittable_level_count": 33,
        "count_delta": 0,
        "ready_for_operator_submit": False,
    }
    null_artifacts["B1"] = {
        **_b1(),
        "live_submittable_level_count": 33,
        "refreshed_package_path": "",
    }
    null_artifact = mod.build_artifact(
        artifacts=null_artifacts,
        live_flags_by_name=_live_flags(),
        preconditions_checked={"ok": False},
        duration_s=0.001,
    )
    assert null_artifact["honest_verdict"].endswith("_preconditions")

    run_artifact = mod.run(tmp_path, live_flags_by_name={}, write=False)
    assert run_artifact["duration_s"] >= 0.0001

    monkeypatch.setattr(mod, "validate_artifact", lambda artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.run(tmp_path, live_flags_by_name={}, write=False, duration_s=0.001)
