"""Tests for Exp 4289 .396 capstone aggregation.

Spec refs: REQ-CAPSTONE-4289, SCENARIO-CAPSTONE-4289.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v396_4289 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _paper_gate(paper_ready: bool = True) -> JsonDict:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
        "note": "fixture publication gate",
    }


def _minimal_payloads() -> dict[str, JsonDict]:
    return {
        "4281_diffusiongemma": {
            "honest_verdict": "complete: learned guidance beats RFG fixture",
            "diffusiongemma_guidance_moat": True,
            "carnot_minus_rfg_delta": 0.24,
            "guidance_moat_ci95": [0.08, 0.39],
            "guidance_changes_selection": True,
            "verifier_is_oracle": False,
            "headline_arm": {
                "status": "complete",
                "n_completed": 30,
                "verifier_is_oracle": False,
            },
            "execution_grounded_arm": {
                "status": "not_headlined",
                "execution_grounded_guidance_delta": 0.18,
                "verifier_is_oracle": True,
            },
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4282_arcgen": {
            "honest_verdict": "complete: arcgen cross-family fixture",
            "arcgen_cross_family_holds": True,
            "cross_family_delta": 0.31,
            "cross_family_ci95": [0.12, 0.5],
            "per_substrate_delta": {
                "arcgen": {"cross_family_delta": 0.31},
                "original_arc": {"cross_family_delta": 0.4},
            },
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4283_self_learning": {
            "honest_verdict": "complete: online adaptation helps fixture",
            "online_adaptation_helps": True,
            "static_cross_family_delta": 0.2,
            "online_cross_family_delta": 0.34,
            "online_minus_static_ci95": [0.03, 0.23],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4284_efficiency": {
            "honest_verdict": "complete: efficiency fixture",
            "efficiency_parity_at_lower_cost": True,
            "cost_ratio": 0.05,
            "accuracy_energy_verifier": 0.7,
            "accuracy_llm_judge": 0.69,
            "accuracy_delta": 0.01,
            "accuracy_delta_ci95": [-0.03, 0.05],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4285_arc_progress": {
            "honest_verdict": "success: arc progress fixture",
            "total_levels": 23,
            "total_levels_solved": 23,
            "levels_completed": 2,
            "new_levels_solved_this_task": 2,
            "game_advanced": "fixture-game",
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4287_registry": {
            "honest_verdict": "complete: registry fixture",
            "registry_reconciled": True,
            "regression_guard_passed": True,
            "gaps_logged": [{"gap_id": "fixture-gap"}],
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4288_hardware": {
            "honest_verdict": "complete: hardware fixture",
            "kv260_terminal_confirmed": False,
            "kv260_step_taken": "kv260_fixture_step",
            "polarfire_step_taken": "polarfire_fixture_step",
            "gatemate_step_taken": "gatemate_fixture_step",
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def _expect_validation_error(artifact: JsonDict, mutator: Any, match: str) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)


def test_req_capstone_4289_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4289: OpenSpec declares the .396 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4289" in spec
    assert "SCENARIO-CAPSTONE-4289" in spec
    assert "experiment_4289_capstone_v396.json" in spec
    assert "blocked_v396_artifacts_missing" in spec
    assert "publication_gate.py --json" in spec
    assert "flagged_adversarial:true" in spec
    assert "verifier_is_oracle:false" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4289_current_artifacts_report_honest_v396_read() -> None:
    """SCENARIO-CAPSTONE-4289: current .396 artifacts produce the honest headline."""

    artifact = mod.build_artifact(Path.cwd(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == (
        "partial_state_blocked_arcgen_excluded_flagged_efficiency_parity_"
        "self_learning_excluded_flagged_arc21_game_ls20-9607627b_paper_ready"
    )
    assert artifact["honest_verdict"] == (
        "complete: diffusiongemma_partial_state_blocked_arcgen_excluded_flagged_"
        "efficiency_parity_arc21"
    )
    assert artifact["guidance_moat_holds"] is False
    assert artifact["cross_family_hardens_on_arcgen"] is False
    assert artifact["verifier_efficiency_parity"] is True
    assert artifact["diffusiongemma_thesis_state"] == "partial_state_blocked"
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["verifier_is_oracle_honored"] is True

    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4282_arcgen", "4283_self_learning"}
    assert artifact["diffusiongemma_guidance"]["carnot_minus_rfg_delta"] == pytest.approx(0.0)
    assert artifact["diffusiongemma_guidance"]["guidance_moat_ci95"] == [0.0, 0.0]
    assert artifact["diffusiongemma_guidance"]["headline_arm_verifier_is_oracle"] is False
    assert artifact["diffusiongemma_guidance"]["execution_grounded_arm"]["moat_eligible"] is False
    assert artifact["arcgen_cross_family"]["status"] == "excluded_flagged_adversarial"
    assert "cross_family_delta" not in artifact["arcgen_cross_family"]
    assert artifact["self_learning"]["status"] == "excluded_flagged_adversarial"
    assert "online_cross_family_delta" not in artifact["self_learning"]
    assert artifact["efficiency"]["cost_ratio"] == pytest.approx(1.95e-8)
    assert artifact["arc_progress"]["total_levels"] == 21
    assert artifact["arc_progress"]["game_advanced"] == "ls20-9607627b"

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAMS)
    for key, upstream in mod.DEFAULT_UPSTREAMS.items():
        expected_sha = hashlib.sha256((Path.cwd() / upstream.path).read_bytes()).hexdigest()
        assert provenance[key]["sha256"] == expected_sha
    assert provenance["4282_arcgen"]["fields_imported"] == []
    assert provenance["4283_self_learning"]["fields_imported"] == []
    assert "cost_ratio" in provenance["4284_efficiency"]["fields_imported"]
    assert artifact["reproducibility_checksum"] == mod.checksum_from_provenance(
        artifact["upstream_provenance"]
    )


def test_req_capstone_4289_clean_fixture_reports_external_verifier_win(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4289: clean oracle-distinct inputs can headline the §5 moat."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == (
        "external_verifier_improves_generation_arcgen_hardens_efficiency_parity_"
        "self_learning_helps_arc23_game_fixture-game_paper_not_ready"
    )
    assert artifact["guidance_moat_holds"] is True
    assert artifact["cross_family_hardens_on_arcgen"] is True
    assert artifact["verifier_efficiency_parity"] is True
    assert artifact["diffusiongemma_thesis_state"] == "external_verifier_improves_generation"
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["arcgen_cross_family"]["cross_family_delta"] == pytest.approx(0.31)
    assert artifact["self_learning"]["online_adaptation_helps"] is True
    assert artifact["verifier_is_oracle_honored"] is True


def test_req_capstone_4289_excludes_flagged_and_blocks_circular_headlines(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4289: flagged or circular inputs cannot headline a moat."""

    payloads = _minimal_payloads()
    payloads["4281_diffusiongemma"]["headline_arm"]["verifier_is_oracle"] = True
    payloads["4281_diffusiongemma"]["verifier_is_oracle"] = True
    payloads["4281_diffusiongemma"]["execution_grounded_arm"] = {
        "status": "complete",
        "execution_grounded_guidance_delta": 0.2,
        "verifier_is_oracle": True,
    }
    payloads["4282_arcgen"]["flagged_adversarial"] = True
    payloads["4284_efficiency"]["verifier_is_oracle"] = True
    _write_default_artifacts(tmp_path, payloads)

    def live_flags(path: Path) -> list[dict[str, str]]:
        if path.name == mod.DEFAULT_UPSTREAMS["4283_self_learning"].path.name:
            return [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}]
        return []

    artifact = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.25,
        live_flag_runner=live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["guidance_moat_holds"] is False
    assert artifact["cross_family_hardens_on_arcgen"] is False
    assert artifact["verifier_efficiency_parity"] is False
    assert artifact["diffusiongemma_thesis_state"] == "execution_grounded_only"
    assert artifact["verifier_is_oracle_honored"] is False
    assert artifact["oracle_distinct_violations"] == [
        "4281_diffusiongemma:headline_guidance",
        "4284_efficiency:efficiency",
    ]
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4282_arcgen", "4283_self_learning"}
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4282_arcgen"]["fields_imported"] == []
    assert provenance["4283_self_learning"]["fields_imported"] == []


def test_req_capstone_4289_missing_or_malformed_core_blocks_without_aggregation(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4289: missing or unparsable core artifacts produce blocked verdicts."""

    payloads = _minimal_payloads()
    payloads.pop("4284_efficiency")
    _write_default_artifacts(tmp_path, payloads)
    malformed_path = tmp_path / mod.DEFAULT_UPSTREAMS["4288_hardware"].path
    malformed_path.write_text("[]\n", encoding="utf-8")

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v396_artifacts_missing"
    assert artifact["headline_outcome"] == "blocked_v396_artifacts_missing"
    assert artifact["guidance_moat_holds"] is False
    assert artifact["cross_family_hardens_on_arcgen"] is False
    assert artifact["verifier_efficiency_parity"] is False
    assert artifact["diffusiongemma_thesis_state"] == "partial_state_blocked"
    assert artifact["paper_ready"] is None
    assert artifact["missing_upstream_artifacts"] == [
        {"artifact_key": "4284_efficiency", "experiment_id": 4284, "reason": "missing"},
        {
            "artifact_key": "4288_hardware",
            "experiment_id": 4288,
            "reason": "unparsable_or_non_object",
        },
    ]
    assert artifact["upstream_provenance"] == []
    assert artifact["reproducibility_checksum"] == mod.BLOCKED_CHECKSUM


def test_req_capstone_4289_validation_write_and_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4289: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4289_capstone_v396.json"),
        started_s=6.0,
        now_s=6.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)

    assert mod.bool_metric({"x": True}, "x") is True
    assert mod.bool_metric({"x": 1}, "x") is None
    assert mod.int_metric({"x": 2}, "x") == 2
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": 2}, "x") == pytest.approx(2.0)
    assert mod.float_metric({"x": True}, "x") is None
    assert mod.str_metric({"x": "ok"}, "x") == "ok"
    assert mod.str_metric({"x": 1}, "x") == ""
    assert mod.list_metric({"x": [1]}, "x") == [1]
    assert mod.list_metric({"x": "bad"}, "x") == []
    assert mod.dict_metric({"x": {"y": 1}}, "x") == {"y": 1}
    assert mod.dict_metric({"x": []}, "x") == {}
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "sha256:" + "a" * 64}) == (
        "a" * 64
    )
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "b" * 64}) == "b" * 64
    assert mod.sha_from_payload_checksum({}) == ""
    assert mod.live_has_critical([{"severity": "critical"}]) is True
    assert mod.live_has_critical([{"severity": "warn"}]) is False
    assert mod.clean_payload(None, False) is None
    assert mod.clean_payload({"x": 1}, True) is None
    assert mod.clean_payload({"x": 1}, False) == {"x": 1}
    assert mod.checksum_from_provenance([]) == mod.BLOCKED_CHECKSUM
    assert mod._ci_excludes_zero({"x": [1.0]}, "x") is False
    assert mod._ci_excludes_zero({"x": ["bad", 1.0]}, "x") is False
    assert mod._ci_excludes_zero({"x": [1.0, "bad"]}, "x") is False
    assert mod._ci_excludes_zero({"x": [-2.0, -1.0]}, "x") is True
    assert mod.guidance_read(
        {
            "diffusiongemma_guidance_moat": True,
            "per_arm_verifier_is_oracle": {"headline_learned": False},
        },
        skipped=False,
    )["guidance_moat_holds"] is True
    assert mod.guidance_read(
        {"diffusiongemma_guidance_moat": True, "verifier_is_oracle": False},
        skipped=False,
    )["headline_arm_verifier_is_oracle"] is False
    assert mod.guidance_read(
        {"execution_grounded_guidance_delta": 0.4}, skipped=False
    )["execution_grounded_arm"]["execution_grounded_guidance_delta"] == pytest.approx(0.4)
    assert mod.guidance_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.guidance_read(None, skipped=False)["status"] == "missing_or_excluded"
    assert mod.arcgen_read(None, skipped=False)["status"] == "missing_or_excluded"
    assert mod.self_learning_read(None, skipped=False)["status"] == "missing_or_excluded"
    assert mod.self_learning_read({}, skipped=False)["status"] == "static_ceiling"
    assert mod.self_learning_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.efficiency_read(None, skipped=False)["status"] == "missing_or_excluded"
    assert mod.efficiency_read({}, skipped=False)["status"] == "not_parity"
    assert mod.efficiency_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.arc_progress_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.arc_progress_read(None, skipped=False)["status"] == "missing_or_excluded"
    assert mod.registry_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.registry_read(None, skipped=False)["status"] == "missing_or_excluded"
    assert mod.hardware_read({}, skipped=True)["status"] == "excluded_flagged_adversarial"
    assert mod.hardware_read(None, skipped=False)["status"] == "missing_or_excluded"
    assert mod.diffusiongemma_thesis_state({"guidance_moat_holds": False}) == (
        "ties_model_self_guidance"
    )
    assert mod._status_part("other", "clean", "excluded", "fallback") == "fallback"
    assert mod._headline_outcome(
        "ties_model_self_guidance",
        {"status": "does_not_harden"},
        {"verifier_efficiency_parity": False},
        {"status": "static_ceiling"},
        {"total_levels": 5, "game_advanced": "g"},
        False,
    ) == "ties_model_self_guidance_arcgen_not_hardened_efficiency_not_parity_self_learning_static_ceiling_arc5_game_g_paper_not_ready"
    assert mod._oracle_violations(
        {"reported_diffusiongemma_guidance_moat": False},
        {"reported_arcgen_cross_family_holds": True, "verifier_is_oracle": True},
        {"reported_efficiency_parity_at_lower_cost": False},
    ) == ["4282_arcgen:cross_family"]
    assert mod.is_sha256("a" * 64) is True
    assert mod.is_sha256("z" * 64) is False
    assert mod.is_sha256(1) is False

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": "[]\n", "returncode": 0})(),
    )
    with pytest.raises(ValueError, match="publication_gate.py --json"):
        mod.run_publication_gate(tmp_path)

    _expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    _expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    _expect_validation_error(artifact, lambda a: a.update({"headline_outcome": ""}), "headline")
    _expect_validation_error(
        artifact,
        lambda a: a.update({"guidance_moat_holds": "yes"}),
        "guidance_moat_holds",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"diffusiongemma_thesis_state": "bad"}),
        "diffusiongemma_thesis_state",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"flagged_artifacts_excluded": "bad"}),
        "flagged_artifacts_excluded",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"reproducibility_checksum": "bad"}),
        "reproducibility_checksum",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"paper_ready": "yes"}),
        "paper_ready",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"upstream_provenance": "bad"}),
        "upstream_provenance",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"field_principles": {}}),
        "field_principles",
    )
    _expect_validation_error(
        artifact,
        lambda a: a.update({"reproducibility_checksum": "0" * 64}),
        "does not match",
    )

    def corrupt_row_sha(a: JsonDict) -> None:
        a["upstream_provenance"][0]["sha256"] = "bad"
        a["reproducibility_checksum"] = mod.checksum_from_provenance(a["upstream_provenance"])

    _expect_validation_error(artifact, corrupt_row_sha, "invalid sha256")

    def corrupt_skipped_imports(a: JsonDict) -> None:
        a["upstream_provenance"][0]["skipped"] = True
        a["upstream_provenance"][0]["fields_imported"] = ["x"]

    _expect_validation_error(artifact, corrupt_skipped_imports, "skipped upstreams")

    blocked = mod._blocked_artifact(
        [{"artifact_key": "x", "experiment_id": 1, "reason": "missing"}],
        started_s=0.0,
        now_s=1.0,
    )
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact({**blocked, "paper_ready": True})
