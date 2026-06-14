"""Tests for Exp 4173 .386 capstone aggregation.

Spec refs: REQ-CAPSTONE-4173, SCENARIO-CAPSTONE-4173.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v386_4173 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[int, JsonDict]:
    return {
        4165: {
            "schema": "carnot.capstone_v385_4165.v1",
            "headline_outcome": "baseline_advancing_moat_rerank_confirmed",
            "total_games_solved": 13,
            "honest_verdict": "blocked: capstone_v385_accumulation_still_blocked",
        },
        4167: {
            "schema": "carnot.experiment_4167_outerloop_training_monitor.v1",
            "current_val_exact_accuracy": 0.504166662693,
            "checkpoint_mtime": "2026-06-13T04:41:29.093138Z",
            "outerloop_train_alive": True,
            "baseline_faithful": False,
            "val_crossed_085": False,
            "honest_verdict": "complete: outerloop_training_alive_val_0.5042_below_0.85",
        },
        4168: {
            "schema": "carnot.experiment_4168_decisive_verifier_graft_defensive.v1",
            "graft_deferred": True,
            "verifier_value_added": False,
            "baseline_status": {"current_val_exact_accuracy": 0.514843761921},
            "honest_verdict": "complete: graft_deferred_outerloop_training_val_0.5148",
        },
        4169: {
            "flagged_adversarial": True,
            "total_games_solved": 99,
            "honest_verdict": "complete: flagged_arc_should_not_import",
        },
        4170: {
            "flagged_for_v387": "vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387",
            "methods_mapped": [{"name": "TRM nano-trm baseline and headroom gate"}],
            "honest_verdict": "complete: sota_ingestion_verifier_moat_guidance_mapped_v387",
        },
        4171: {
            "regression_guard_passed": True,
            "diffusiongemma_gate_state": {
                "state": "kept_gated",
                "graft_deferred": True,
                "verifier_value_added": False,
                "rerank_ci_excludes_zero_positive": False,
            },
            "honest_verdict": "complete: registry_gaps_reconciled_to_v386_truth",
        },
        4172: {
            "kv260_terminal_confirmed": True,
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "honest_verdict": "complete: hardware_continuity_4172",
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4173_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4173: OpenSpec declares headlines, principles, and skip rules."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4173" in spec
    assert "SCENARIO-CAPSTONE-4173" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "flagged_adversarial:true" in spec
    assert "0.278 seed" in spec


def test_scenario_capstone_4173_current_artifacts_form_decision_grade_verdict() -> None:
    """SCENARIO-CAPSTONE-4173: current .386 artifacts aggregate without flagged metrics."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=10.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "outerloop_training_in_progress"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v386_outerloop_training_in_progress_"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["headline_answers"] == {
        "outerloop_advanced_past_0278_toward_087": True,
        "defensive_graft_result": "deferred",
        "diffusiongemma_gate_status": "STILL-PENDING",
        "total_arc_games_solved": 13,
    }

    trajectory = artifact["baseline_val_trajectory"]
    assert trajectory["status"] == "outerloop_training_in_progress"
    assert trajectory["current_val_exact_accuracy"] == pytest.approx(0.504166662693)
    assert trajectory["checkpoint_mtime"] == "2026-06-13T04:41:29.093138Z"
    assert trajectory["advanced_past_seed"] is True
    assert trajectory["val_crossed_087"] is False
    assert trajectory["outerloop_train_alive"] is True

    assert artifact["defensive_graft_verdict"]["status"] == "deferred"
    assert artifact["defensive_graft_verdict"]["verifier_value_added"] is False
    assert artifact["diffusiongemma_gate_status"] == "STILL-PENDING"
    assert artifact["total_arc_games_solved"] == 13
    assert artifact["arc_games_solved"]["source"] == "prior_clean_carry_forward"
    assert artifact["arc_games_solved"]["current_arc_status"] == "skipped_flagged_adversarial"
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 4169,
            "path": "results/experiment_4169_arc_incremental_progress.json",
            "reason": "flagged_adversarial:true",
            "sha256": hashlib.sha256(
                Path("results/experiment_4169_arc_incremental_progress.json").read_bytes()
            ).hexdigest(),
        }
    ]

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    for experiment_id, row in provenance.items():
        expected_sha = hashlib.sha256(
            mod.DEFAULT_UPSTREAM_PATHS[experiment_id].read_bytes()
        ).hexdigest()
        assert row["sha256"] == expected_sha
    assert provenance[4169]["skipped"] is True
    assert provenance[4169]["fields_imported"] == []
    assert "total_games_solved" in provenance[4165]["fields_imported"]
    assert "current_val_exact_accuracy" in provenance[4167]["fields_imported"]
    assert "graft_deferred" in provenance[4168]["fields_imported"]
    assert "diffusiongemma_gate_state" in provenance[4171]["fields_imported"]


def test_req_capstone_4173_clean_branches_for_arc_and_graft_outcomes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4173: clean inputs distinguish progress, validated, null, and stalled."""

    payloads = _minimal_payloads()
    payloads[4169]["flagged_adversarial"] = False
    payloads[4169]["total_games_solved"] = 14
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "outerloop_training_in_progress"
    assert artifact["arc_games_solved"]["source"] == "current_clean_exp4169"
    assert artifact["total_arc_games_solved"] == 14

    payloads = _minimal_payloads()
    payloads[4167]["current_val_exact_accuracy"] = 0.872
    payloads[4167]["outerloop_train_alive"] = False
    payloads[4167]["baseline_faithful"] = True
    payloads[4167]["val_crossed_085"] = True
    payloads[4168]["graft_deferred"] = False
    payloads[4168]["verifier_value_added"] = True
    _write_default_artifacts(tmp_path, payloads)
    validated = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert validated["headline_outcome"] == "baseline_converged_graft_validated"
    assert validated["diffusiongemma_gate_status"] == "RESOLVED-positive"

    payloads[4168]["verifier_value_added"] = False
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert null["headline_outcome"] == "baseline_converged_graft_null"
    assert null["diffusiongemma_gate_status"] == "RESOLVED-null"

    payloads[4167]["current_val_exact_accuracy"] = 0.2
    payloads[4167]["baseline_faithful"] = False
    _write_default_artifacts(tmp_path, payloads)
    stalled = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert stalled["headline_outcome"] == "baseline_stalled"
    assert stalled["baseline_val_trajectory"]["status"] == "baseline_stalled"


def test_req_capstone_4173_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4173: validation protects required fields, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4173_capstone_v386.json"),
        started_s=6.0,
        now_s=6.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_4172 = _minimal_payloads()
    payloads_without_4172.pop(4172)
    _write_default_artifacts(missing_root, payloads_without_4172)
    missing = mod.build_artifact(missing_root, started_s=7.0, now_s=7.25)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4172}]
    assert missing["hardware_continuity"]["status"] == "missing"

    missing_4167 = tmp_path / "missing_4167"
    payloads_without_4167 = _minimal_payloads()
    payloads_without_4167.pop(4167)
    _write_default_artifacts(missing_4167, payloads_without_4167)
    no_baseline = mod.build_artifact(missing_4167, started_s=8.0, now_s=8.25)
    assert no_baseline["baseline_val_trajectory"]["status"] == "missing"
    assert no_baseline["headline_outcome"] == "baseline_stalled"

    assert mod.baseline_val_trajectory({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.defensive_graft_verdict({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.defensive_graft_verdict(None, was_skipped=False)["status"] == "missing"
    assert mod.arc_games_solved(None, {"total_arc_games_solved": 12}, current_was_skipped=False) == {
        "status": "included_carry_forward",
        "source": "prior_clean_carry_forward",
        "source_experiment_id": 4165,
        "current_arc_experiment_id": 4169,
        "current_arc_status": "missing",
        "total_arc_games_solved": 12,
    }
    assert mod.sota_guidance({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.sota_guidance(None, was_skipped=False)["status"] == "missing"
    assert mod.registry_status({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_status(None, was_skipped=False)["status"] == "missing"
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )

    expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    expect_validation_error(artifact, lambda a: a.update({"headline_outcome": "bad"}), "headline")
    expect_validation_error(artifact, lambda a: a.update({"baseline_val_trajectory": []}), "baseline")
    expect_validation_error(
        artifact,
        lambda a: a["baseline_val_trajectory"].update({"current_val_exact_accuracy": "bad"}),
        "current validation",
    )
    expect_validation_error(
        artifact,
        lambda a: a.update({"defensive_graft_verdict": {"status": "bad"}}),
        "defensive graft",
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
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4165"}),
        "integer experiment_id",
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"fields_imported": "bad"}),
        "fields_imported",
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][2].update(
            {"skipped": True, "fields_imported": ["total_games_solved"]}
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

    output_path = tmp_path / "results" / "experiment_4173_capstone_v386.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out


def expect_validation_error(
    artifact: JsonDict, mutator: Any, match: str
) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)
