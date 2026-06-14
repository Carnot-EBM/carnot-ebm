"""Tests for Exp 4183 .387 capstone aggregation.

Spec refs: REQ-CAPSTONE-4183, SCENARIO-CAPSTONE-4183.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v387_4183 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[int, JsonDict]:
    return {
        4175: {
            "honest_verdict": "complete: headroom fixture",
            "headroom_present_domain": "code",
            "max_selectable_headroom": 0.18,
            "per_domain_headroom": {"code": {"selectable_headroom": 0.18}},
        },
        4177: {
            "honest_verdict": "complete: positive moat fixture",
            "domain": "code",
            "verifier_value_added": True,
            "positive_control_confirmed": True,
            "moat_delta_vs_vote": {"delta": 0.18, "ci95": [0.08, 0.3], "status": "measured"},
            "moat_vs_matched_control": {"delta": 0.18, "status": "measured"},
            "accuracy_cost_pareto": {"value_added_basis": "accuracy_lift_ci95_excludes_zero"},
            "positive_control": {"oracle_at_k": 0.9, "sc_vote_pass1": 0.72},
        },
        4178: {
            "honest_verdict": "complete: gap3 bounded fixture",
            "pass2_energy_vs_vote": 0.0,
            "all_four_gates_pass": False,
            "gates": {"selection_pass2_beats_vote": False},
            "headroom_capture_fraction": 0.0,
            "candidate_auroc": 0.893651,
        },
        4179: {
            "honest_verdict": "success: arc fixture",
            "total_levels_solved": 14,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 1,
            "real_env_confirmed": True,
        },
        4180: {
            "honest_verdict": "complete: sota fixture",
            "flagged_for_v388": "cem_gap3_stage2_compositional_arc_energy_v388",
        },
        4181: {
            "honest_verdict": "complete: registry fixture",
            "regression_guard_passed": True,
        },
        4182: {
            "honest_verdict": "complete: hardware fixture",
            "kv260_terminal_confirmed": True,
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4183_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4183: OpenSpec declares statuses, fields, and gate semantics."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4183" in spec
    assert "SCENARIO-CAPSTONE-4183" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for status in mod.VERIFIER_MOAT_STATUSES | mod.DIFFUSIONGEMMA_GATE_STATUSES:
        assert status in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "flagged_adversarial:true" in spec
    assert "verifier_value_added:true" in spec
    assert "positive_control_confirmed:true" in spec


def test_scenario_capstone_4183_current_artifacts_form_decision_grade_verdict() -> None:
    """SCENARIO-CAPSTONE-4183: current clean .387 artifacts aggregate honestly."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=10.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "moat_proven_headroom_present"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v387_moat_proven_headroom_present_"
    )
    assert artifact["verifier_moat_status"] == "PROVEN-headroom-present"
    assert artifact["gap3_stage1_status"] == "BOUNDED"
    assert artifact["diffusiongemma_gate_status"] == "MET"
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["headline_answers"] == {
        "headroom_controlled_moat_verifier_value_added": True,
        "headroom_controlled_moat_positive_control_confirmed": True,
        "headroom_controlled_moat_domain": "code",
        "gap3_stage1_reaches_13pp_headroom": False,
        "gap3_pass2_energy_vs_vote": 0.0,
        "gap3_all_four_gates_pass": False,
        "total_arc_levels_solved": 14,
        "strongest_sota_method_flagged_for_v388": (
            "cem_gap3_stage2_compositional_arc_energy_v388"
        ),
    }

    assert artifact["verifier_moat"]["domain"] == "code"
    assert artifact["verifier_moat"]["headroom_present"] is True
    assert artifact["verifier_moat"]["verifier_value_added"] is True
    assert artifact["gap3_stage1"]["reaches_proven_arc_headroom"] is False
    assert artifact["arc_progress"]["total_arc_levels_solved"] == 14
    assert artifact["sota_v388"]["flagged_for_v388"] == (
        "cem_gap3_stage2_compositional_arc_energy_v388"
    )
    assert artifact["flagged_artifacts_skipped"] == []

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    for experiment_id, rel_path in mod.DEFAULT_UPSTREAM_PATHS.items():
        expected_sha = hashlib.sha256(rel_path.read_bytes()).hexdigest()
        assert provenance[experiment_id]["sha256"] == expected_sha
        assert provenance[experiment_id]["skipped"] is False
    assert "verifier_value_added" in provenance[4177]["fields_imported"]
    assert "pass2_energy_vs_vote" in provenance[4178]["fields_imported"]
    assert "total_levels_solved" in provenance[4179]["fields_imported"]
    assert "flagged_for_v388" in provenance[4180]["fields_imported"]


def test_req_capstone_4183_gate_branches_and_flagged_skips(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4183: gate branches distinguish positive, null, no-headroom, and skipped."""

    payloads = _minimal_payloads()
    payloads[4177]["verifier_value_added"] = False
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    mod.validate_artifact(null)
    assert null["headline_outcome"] == "moat_bounded_clean_null"
    assert null["verifier_moat_status"] == "BOUNDED-clean-null"
    assert null["diffusiongemma_gate_status"] == "STILL-PENDING-headroom-present-null"

    payloads = _minimal_payloads()
    payloads[4175]["headroom_present_domain"] = ""
    payloads[4175]["max_selectable_headroom"] = 0.0
    payloads[4177]["verifier_value_added"] = False
    payloads[4177]["positive_control_confirmed"] = False
    payloads[4178]["pass2_energy_vs_vote"] = 0.13
    payloads[4178]["all_four_gates_pass"] = True
    _write_default_artifacts(tmp_path, payloads)
    reaches = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert reaches["headline_outcome"] == "gap3_reaches_headroom"
    assert reaches["verifier_moat_status"] == "DEFERRED-no-headroom"
    assert reaches["gap3_stage1_status"] == "REACHES-headroom"
    assert reaches["diffusiongemma_gate_status"] == "STILL-PENDING-no-headroom"

    payloads[4178]["pass2_energy_vs_vote"] = 0.0
    payloads[4178]["all_four_gates_pass"] = False
    _write_default_artifacts(tmp_path, payloads)
    deferred = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert deferred["headline_outcome"] == "moat_deferred_no_headroom"

    payloads = _minimal_payloads()
    payloads[4177]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)
    skipped = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert skipped["headline_outcome"] == "gap3_bounded"
    assert skipped["diffusiongemma_gate_status"] == "STILL-PENDING-no-headroom"
    assert skipped["upstream_provenance"][1]["experiment_id"] == 4177
    assert skipped["upstream_provenance"][1]["fields_imported"] == []
    assert skipped["upstream_provenance"][1]["skipped"] is True
    assert skipped["flagged_artifacts_skipped"][0]["experiment_id"] == 4177


def test_req_capstone_4183_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4183: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4183_capstone_v387.json"),
        started_s=6.0,
        now_s=6.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_4182 = _minimal_payloads()
    payloads_without_4182.pop(4182)
    _write_default_artifacts(missing_root, payloads_without_4182)
    missing = mod.build_artifact(missing_root, started_s=7.0, now_s=7.25)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4182}]
    assert missing["hardware_continuity"]["status"] == "missing"

    assert mod.headroom_census({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.list_float_metric({"values": "not-a-list"}, "values") == []
    assert mod.moat_verdict({}, {}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.gap3_stage1({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_progress({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.sota_v388({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_gap_hygiene({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )

    expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    expect_validation_error(artifact, lambda a: a.update({"headline_outcome": "bad"}), "headline")
    expect_validation_error(
        artifact, lambda a: a.update({"verifier_moat_status": "bad"}), "verifier moat"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"gap3_stage1_status": "bad"}), "GAP-3"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"diffusiongemma_gate_status": "bad"}), "DiffusionGemma"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"inference_substrate": "bad"}), "inference_substrate"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"field_principles": []}), "field_principles"
    )
    expect_validation_error(
        artifact, lambda a: a["field_principles"].update({"honest_verdict": "wrong"}), "principle"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"upstream_provenance": {}}), "upstream_provenance"
    )
    expect_validation_error(
        artifact, lambda a: a["upstream_provenance"].append(42), "entries must be objects"
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4175"}),
        "integer experiment_id",
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"fields_imported": "bad"}),
        "fields_imported",
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][1].update(
            {"skipped": True, "fields_imported": ["verifier_value_added"]}
        ),
        "skipped upstreams",
    )
    expect_validation_error(
        artifact, lambda a: a["upstream_provenance"][0].update({"sha256": "bad"}), "sha256"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"flagged_artifacts_skipped": {}}), "flagged"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"reproducibility_checksum": "bad"}), "checksum"
    )
    expect_validation_error(artifact, lambda a: a.update({"duration_s": 99.0}), "checksum")

    output_path = tmp_path / "results" / "experiment_4183_capstone_v387.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out


def expect_validation_error(artifact: JsonDict, mutator: Any, match: str) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)
