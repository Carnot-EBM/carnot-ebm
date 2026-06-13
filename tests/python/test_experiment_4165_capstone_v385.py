"""Tests for Exp 4165 .385 capstone aggregation.

Spec refs: REQ-CAPSTONE-4165, SCENARIO-CAPSTONE-4165.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v385_4165 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4157: {
            "honest_verdict": "complete: contiguous_run_advanced_baseline",
            "flagged_adversarial": False,
            "current_val": 0.501041650772,
            "max_val": 0.501041650772,
            "run_alive": False,
            "baseline_faithful": False,
            "manual_lr_step": 21304,
            "estimated_passes_to_085": {"estimated_additional_val_intervals": 48},
            "val_trajectory": [
                {"val_exact_accuracy": 0.420000001},
                {"val_exact_accuracy": 0.501041650772},
            ],
        },
        4158: {
            "honest_verdict": "complete: verifier_rerank_moat_ci95_excludes_zero_at_val_0.50",
            "flagged_adversarial": False,
            "headroom_present": True,
            "oracle_at_k": 0.625,
            "vote_at_1": 0.5,
            "verifier_recovers_outvoted": 8,
            "rerank_lift_vs_vote": {
                "delta": 0.078125,
                "ci95": [0.015625, 0.125],
                "status": "ci95_excludes_zero_positive",
                "n_puzzles": 64,
                "verifier_pass_at_1": 0.578125,
            },
            "cost_ratio_vs_llm_judge": {"ratio": 23158.363},
        },
        4159: {
            "honest_verdict": "complete: graft_deferred_baseline_below_0.85",
            "flagged_adversarial": False,
            "graft_deferred": True,
            "verifier_value_added": False,
            "current_val": 0.501041650772,
            "rft_vs_ablation_delta": {
                "delta": 0.0,
                "ci95": [0.0, 0.0],
                "status": "deferred_baseline_below_0.85",
                "n_matched": 0,
            },
        },
        4160: {
            "honest_verdict": "complete: verifier_pruner_1.31x_action_efficient",
            "action_efficiency_ratio": 1.3125,
            "baseline_actions": 21,
            "verifier_actions": 16,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 0,
            "real_env_confirmed": False,
        },
        4161: {
            "honest_verdict": "complete: observability_timing_detector_fallback_landed",
            "fix_applied": True,
            "fallback_added": True,
            "research_conductor_touched": False,
        },
        4162: {
            "honest_verdict": "complete: sota_ingestion_verifier_moat_guidance_mapped",
            "flagged_for_v386": "entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386",
            "methods_mapped": [{"name": "EntRGi", "url": "https://arxiv.org/abs/2602.05000"}],
        },
        4163: {
            "honest_verdict": "complete: registry_gaps_reconciled_to_v385_truth",
            "flagged_adversarial": False,
            "regression_guard_passed": True,
            "diffusiongemma_gate_state": {
                "state": "kept_gated",
                "moved_by_rerank_signal": False,
                "rerank_ci_excludes_zero_positive": False,
                "verifier_value_added": False,
            },
        },
        4164: {
            "honest_verdict": "complete: hardware_continuity_4164",
            "kv260_terminal_confirmed": True,
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4165_spec_anchor_declares_artifact_contract() -> None:
    """REQ-CAPSTONE-4165: OpenSpec declares fields, headlines, and skip rules."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4165" in spec
    assert "SCENARIO-CAPSTONE-4165" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "flagged_adversarial:true" in spec
    assert "skipped_flagged_adversarial" in spec


def test_scenario_capstone_4165_current_flagged_inputs_are_provenance_only() -> None:
    """SCENARIO-CAPSTONE-4165: flagged current upstreams are excluded before import."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=10.5)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "accumulation_still_blocked"
    assert artifact["honest_verdict"].startswith(
        "blocked: capstone_v385_accumulation_still_blocked_"
        "baseline_skipped_flagged_adversarial_"
        "rerank_skipped_flagged_adversarial_"
        "graft_skipped_flagged_adversarial_"
        "diffusiongemma_STILL-PENDING_games13_flagged_skipped4"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["baseline_val_trajectory"]["status"] == "skipped_flagged_adversarial"
    assert artifact["baseline_val_trajectory"]["advanced_vs_seed"] is None
    assert artifact["moat_rerank_verdict"]["status"] == "skipped_flagged_adversarial"
    assert artifact["moat_rerank_verdict"]["decision"] == "uninformative"
    assert artifact["graft_verdict"]["status"] == "skipped_flagged_adversarial"
    assert artifact["diffusiongemma_gate_status"] == "STILL-PENDING"
    assert artifact["arc_action_efficiency"]["action_efficiency_ratio"] == pytest.approx(1.3125)
    assert artifact["arc_action_efficiency"]["total_games_solved"] == 13
    assert artifact["observability_status"]["fixed"] is True

    skipped = artifact["flagged_artifacts_skipped"]
    assert [row["experiment_id"] for row in skipped] == [4157, 4158, 4159, 4163]

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    for experiment_id, row in provenance.items():
        expected_sha = hashlib.sha256(
            mod.DEFAULT_UPSTREAM_PATHS[experiment_id].read_bytes()
        ).hexdigest()
        assert row["sha256"] == expected_sha
    for experiment_id in [4157, 4158, 4159, 4163]:
        assert provenance[experiment_id]["fields_imported"] == []
        assert provenance[experiment_id]["skipped"] is True
    assert "action_efficiency_ratio" in provenance[4160]["fields_imported"]
    assert "fix_applied" in provenance[4161]["fields_imported"]
    assert "flagged_for_v386" in provenance[4162]["fields_imported"]
    assert "kv260_terminal_confirmed" in provenance[4164]["fields_imported"]


def test_req_capstone_4165_clean_rerank_positive_decides_intermediate_moat(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4165: clean headroom plus CI>0 confirms the rerank moat."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "baseline_advancing_moat_rerank_confirmed"
    assert artifact["baseline_val_trajectory"]["advanced_vs_seed"] is True
    assert artifact["baseline_val_trajectory"]["run_alive"] is False
    assert artifact["baseline_val_trajectory"]["rounded_values"] == pytest.approx(
        [0.106, 0.278, 0.42, 0.501]
    )
    assert artifact["moat_rerank_verdict"]["status"] == "confirmed"
    assert artifact["moat_rerank_verdict"]["ci95"] == pytest.approx([0.015625, 0.125])
    assert artifact["graft_verdict"]["status"] == "deferred"
    assert artifact["diffusiongemma_gate_status"] == "RESOLVED-positive"


def test_req_capstone_4165_rerank_null_no_headroom_and_graft_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4165: null, no-headroom, faithful graft headlines are distinct."""

    payloads = _clean_payloads()
    payloads[4158]["rerank_lift_vs_vote"]["delta"] = 0.0
    payloads[4158]["rerank_lift_vs_vote"]["ci95"] = [-0.03125, 0.03125]
    payloads[4158]["rerank_lift_vs_vote"]["status"] = "headroom_backed_null_ci95_includes_zero"
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert null["headline_outcome"] == "baseline_advancing_moat_rerank_null"
    assert null["moat_rerank_verdict"]["status"] == "null"
    assert null["diffusiongemma_gate_status"] == "RESOLVED-null"

    payloads = _clean_payloads()
    payloads[4158]["headroom_present"] = False
    payloads[4158]["oracle_at_k"] = 0.5
    payloads[4158]["vote_at_1"] = 0.5
    payloads[4158]["rerank_lift_vs_vote"]["delta"] = 0.0
    payloads[4158]["rerank_lift_vs_vote"]["ci95"] = [0.0, 0.0]
    payloads[4158]["rerank_lift_vs_vote"]["status"] = "no_headroom_oracle_at_k_lte_vote_at_1"
    _write_default_artifacts(tmp_path, payloads)
    no_headroom = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert no_headroom["headline_outcome"] == "baseline_advancing_moat_rerank_no_headroom"
    assert no_headroom["moat_rerank_verdict"]["status"] == "no_headroom"
    assert no_headroom["diffusiongemma_gate_status"] == "STILL-PENDING"

    payloads = _clean_payloads()
    payloads[4157]["current_val"] = 0.86
    payloads[4157]["max_val"] = 0.86
    payloads[4157]["baseline_faithful"] = True
    payloads[4157]["val_trajectory"][-1]["val_exact_accuracy"] = 0.86
    payloads[4159]["graft_deferred"] = False
    payloads[4159]["verifier_value_added"] = True
    payloads[4159]["rft_vs_ablation_delta"] = {
        "delta": 0.04,
        "ci95": [0.01, 0.07],
        "status": "ci95_excludes_zero_positive",
        "n_matched": 64,
    }
    _write_default_artifacts(tmp_path, payloads)
    graft_positive = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert graft_positive["headline_outcome"] == "baseline_faithful_graft_validated"
    assert graft_positive["graft_verdict"]["status"] == "ran_value_added"
    assert graft_positive["diffusiongemma_gate_status"] == "RESOLVED-positive"

    payloads[4159]["verifier_value_added"] = False
    payloads[4159]["rft_vs_ablation_delta"] = {
        "delta": 0.0,
        "ci95": [-0.02, 0.02],
        "status": "ci95_includes_zero",
        "n_matched": 64,
    }
    _write_default_artifacts(tmp_path, payloads)
    graft_null = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)
    assert graft_null["headline_outcome"] == "baseline_faithful_graft_null"
    assert graft_null["graft_verdict"]["status"] == "ran_null"
    assert graft_null["diffusiongemma_gate_status"] == "RESOLVED-null"


def test_req_capstone_4165_validation_and_main_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4165: validation protects headlines, principles, and provenance."""

    _write_default_artifacts(tmp_path, _clean_payloads())
    artifact = mod.build_artifact(tmp_path, started_s=6.0, now_s=6.25)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    assert mod.list_float_metric({"x": "bad"}, "x") == []
    assert mod.trajectory_current_marker({"val_trajectory": "bad"}) is None
    assert mod.trajectory_current_marker({"val_trajectory": [42, {"val_exact_accuracy": 0.1}]}) is None
    assert mod.baseline_val_trajectory(None, was_skipped=False)["status"] == "missing"
    marker_baseline = mod.baseline_val_trajectory(
        {
            "current_val": 0.5,
            "max_val": 0.5,
            "run_alive": True,
            "val_trajectory": [{"val_exact_accuracy": 0.43}],
        },
        was_skipped=False,
    )
    assert marker_baseline["rounded_values"] == pytest.approx([0.106, 0.278, 0.42, 0.43, 0.5])
    assert mod.moat_rerank_verdict(None, was_skipped=False)["status"] == "missing"
    weird_moat = mod.moat_rerank_verdict(
        {
            "headroom_present": True,
            "rerank_lift_vs_vote": {"delta": 0.01, "ci95": [0.01, -0.01]},
        },
        was_skipped=False,
    )
    assert weird_moat["status"] == "null"
    assert mod.graft_verdict(None, was_skipped=False)["status"] == "missing"
    assert mod.arc_action_efficiency({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_action_efficiency(None, was_skipped=False)["status"] == "missing"
    assert mod.observability_status({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.observability_status(None, was_skipped=False)["status"] == "missing"
    assert mod.sota_guidance({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.sota_guidance(None, was_skipped=False)["status"] == "missing"
    assert mod.registry_status(None, was_skipped=False)["status"] == "missing"
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.headline_outcome(
        {"advanced_vs_seed": True, "baseline_faithful": False},
        {"status": "missing"},
        {"status": "missing"},
    ) == "accumulation_still_blocked"

    output_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4165_capstone_v385.json"),
        started_s=8.0,
        now_s=8.25,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    artifact["headline_outcome"] = "not_enumerated"
    with pytest.raises(ValueError, match="headline_outcome"):
        mod.validate_artifact(artifact)

    artifact["headline_outcome"] = "baseline_advancing_moat_rerank_confirmed"
    artifact["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(artifact)

    artifact["upstream_provenance"][0]["sha256"] = "a" * 64
    artifact["field_principles"]["honest_verdict"] = "wrong"
    with pytest.raises(ValueError, match="honest_verdict principle"):
        mod.validate_artifact(artifact)

    artifact["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    artifact["moat_rerank_verdict"]["status"] = "mystery"
    with pytest.raises(ValueError, match="moat_rerank_verdict"):
        mod.validate_artifact(artifact)

    def expect_validation_error(mutator: Any, match: str) -> None:
        mutated = mod.build_artifact(tmp_path, started_s=9.0, now_s=9.25)
        mutator(mutated)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(mutated)

    expect_validation_error(lambda a: a.pop("honest_verdict"), "missing required")
    expect_validation_error(lambda a: a.update({"honest_verdict": "bad"}), "terminal prefix")
    expect_validation_error(lambda a: a.update({"baseline_val_trajectory": {}}), "numeric values")
    expect_validation_error(
        lambda a: a["baseline_val_trajectory"].update({"values": ["bad"]}),
        "values must be numeric",
    )
    expect_validation_error(lambda a: a.update({"graft_verdict": {"status": "bad"}}), "graft_verdict")
    expect_validation_error(lambda a: a.update({"diffusiongemma_gate_status": "bad"}), "diffusiongemma")
    expect_validation_error(lambda a: a.update({"inference_substrate": "bad"}), "inference_substrate")
    expect_validation_error(lambda a: a.update({"field_principles": []}), "field_principles")
    expect_validation_error(lambda a: a.update({"upstream_provenance": {}}), "upstream_provenance")
    expect_validation_error(lambda a: a.update({"upstream_provenance": [42]}), "entries must be objects")
    expect_validation_error(
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4157"}),
        "integer experiment_id",
    )
    expect_validation_error(
        lambda a: a["upstream_provenance"][0].update({"fields_imported": "bad"}),
        "fields_imported must be a list",
    )
    expect_validation_error(
        lambda a: a["upstream_provenance"][0].update(
            {"skipped": True, "fields_imported": ["current_val"]}
        ),
        "skipped upstreams",
    )
    expect_validation_error(
        lambda a: a.update({"flagged_artifacts_skipped": {}}),
        "flagged_artifacts_skipped",
    )
    expect_validation_error(lambda a: a.update({"reproducibility_checksum": "bad"}), "checksum")
    expect_validation_error(lambda a: a.update({"duration_s": 99.0}), "checksum mismatch")

    missing_root = tmp_path / "missing"
    payloads = _clean_payloads()
    payloads.pop(4164)
    _write_default_artifacts(missing_root, payloads)
    missing = mod.build_artifact(missing_root, started_s=7.0, now_s=7.25)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4164}]
    assert {row["experiment_id"] for row in missing["upstream_provenance"]} == set(
        mod.UPSTREAM_IDS
    ) - {4164}

    output = tmp_path / "results" / "experiment_4165_capstone_v385.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output)
    assert mod.main() == 0
    assert str(output) in capsys.readouterr().out
