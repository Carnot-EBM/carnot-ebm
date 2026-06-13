"""Tests for Exp 4155 .384 capstone aggregation.

Spec refs: REQ-CAPSTONE-4155, SCENARIO-CAPSTONE-4155.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v384_4155 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_prior_reference(root: Path) -> None:
    _write_json(
        root / mod.PRIOR_BASELINE_REFERENCE_PATH,
        {
            "experiment_id": 4133,
            "baseline_val_trajectory": {
                "upstream_values": [0.105989582837, 0.278172343969],
                "upstream_deltas": [0.172182761132],
                "final_val_exact_accuracy": 0.278172343969,
            },
        },
    )
    _write_json(
        root / mod.PRIOR_ARC_REFERENCE_PATH,
        {
            "experiment_id": 4144,
            "total_arc_games_solved": 13,
            "total_arc_levels_solved": 13,
        },
    )


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4146: {
            "honest_verdict": "complete: pass1_epoch_advanced_val_0.45",
            "flagged_adversarial": False,
            "seed_epoch": 6399,
            "post_epoch": 6999,
            "val_exact_accuracy": 0.45,
            "max_epochs_cap_confirmed": True,
            "duration_s": 3600.0,
        },
        4147: {
            "honest_verdict": "complete: pass2_val_0.62",
            "flagged_adversarial": False,
            "post_epoch": 7999,
            "val_exact_accuracy": 0.62,
            "native_trainer_launched": True,
            "duration_s": 3600.0,
        },
        4148: {
            "honest_verdict": "complete: pass3_val_0.78",
            "flagged_adversarial": False,
            "post_epoch": 8999,
            "val_exact_accuracy": 0.78,
            "native_trainer_launched": True,
            "duration_s": 3600.0,
        },
        4149: {
            "honest_verdict": "complete: pass4_val_0.872_matches_published_087",
            "flagged_adversarial": False,
            "val_exact_accuracy": 0.872,
            "matches_published_087": True,
            "native_trainer_launched": True,
            "duration_s": 3600.0,
            "val_trajectory_v384": [
                {
                    "experiment": "experiment_4145_archive_v383_activate_v384",
                    "pass_label": "v384_start",
                    "val_exact_accuracy": 0.278172343969,
                    "effective_val_exact_accuracy": 0.278172343969,
                },
                {
                    "experiment": "experiment_4146_sudoku_accumulate_pass1_epochfix",
                    "pass_label": "pass1",
                    "post_epoch": 6999,
                    "val_exact_accuracy": 0.45,
                    "effective_val_exact_accuracy": 0.45,
                },
                {
                    "experiment": "experiment_4147_sudoku_accumulate_pass2",
                    "pass_label": "pass2",
                    "post_epoch": 7999,
                    "val_exact_accuracy": 0.62,
                    "effective_val_exact_accuracy": 0.62,
                },
                {
                    "experiment": "experiment_4148_sudoku_accumulate_pass3",
                    "pass_label": "pass3",
                    "post_epoch": 8999,
                    "val_exact_accuracy": 0.78,
                    "effective_val_exact_accuracy": 0.78,
                },
                {
                    "experiment": "experiment_4149_sudoku_accumulate_pass4_convergence",
                    "pass_label": "pass4",
                    "val_exact_accuracy": 0.872,
                    "effective_val_exact_accuracy": 0.872,
                },
            ],
        },
        4150: {
            "honest_verdict": "success: verifier_value_added_rft_A_gt_B",
            "flagged_adversarial": False,
            "graft_deferred": False,
            "verifier_value_added": True,
            "rerank_lift_vs_vote": {
                "delta": 0.05,
                "ci95": [0.01, 0.09],
                "status": "measured",
            },
            "rft_vs_ablation_delta": {
                "delta": 0.08,
                "ci95": [0.02, 0.14],
                "status": "measured",
                "n_matched": 64,
            },
        },
        4151: {
            "honest_verdict": "complete: sixteenth_game_solved",
            "flagged_adversarial": False,
            "total_games_solved": 14,
            "real_env_confirmed": True,
            "verifier_validated": True,
        },
        4152: {
            "honest_verdict": "complete: sota_ingestion_recursive_reasoner_verifier_energy_guidance_mapped",
            "flagged_for_v385": "diffusiongemma_sedd_verifier_energy_guidance_probe_v385",
            "methods_mapped": [{"name": "DiffusionGemma queued discrete-text substrate"}],
        },
        4153: {
            "honest_verdict": "complete: registry_gaps_reconciled_to_v384_truth",
            "flagged_adversarial": False,
            "regression_guard_passed": True,
            "diffusiongemma_gate_state": {
                "state": "unlocked",
                "verifier_value_added": True,
                "uses_executable_oracle_upper_bound": False,
            },
        },
        4154: {
            "honest_verdict": "complete: hardware_continuity_4154",
            "kv260_terminal_confirmed": True,
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    _write_prior_reference(root)
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4155_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4155: OpenSpec declares the .384 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4155" in spec
    assert "SCENARIO-CAPSTONE-4155" in spec
    assert "baseline_converged_graft_validated" in spec
    assert "accumulation_still_blocked" in spec
    assert "DiffusionGemma" in spec
    assert "sha256" in spec


def test_scenario_capstone_4155_current_artifacts_emit_still_blocked() -> None:
    """SCENARIO-CAPSTONE-4155: current flagged upstreams are provenance-only."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=10.5)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "accumulation_still_blocked"
    assert artifact["honest_verdict"].startswith(
        "blocked: capstone_v384_accumulation_still_blocked_epochfix0_"
        "baseline0870_graft_skipped_flagged_adversarial_diffusiongemma_STILL-PENDING_"
        "games13_flagged_skipped7"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["headline_answers"] == {
        "exp4146_epoch_fix_unstalled": False,
        "exp4149_matches_published_087": None,
        "exp4150_decisive_graft_status": "skipped_flagged_adversarial",
        "diffusiongemma_gate_status": "STILL-PENDING",
        "total_arc_games_solved": 13,
    }
    assert artifact["baseline_val_trajectory"]["status"] == "accumulation_still_blocked"
    assert artifact["baseline_val_trajectory"]["rounded_values"] == pytest.approx([0.106, 0.2782])
    assert artifact["baseline_val_trajectory"]["exp4146_epoch_fix_unstalled"] is False
    assert artifact["baseline_val_trajectory"]["matches_published_087"] is None
    assert artifact["diffusiongemma_gate_status"] == "STILL-PENDING"
    assert artifact["arc_games"]["status"] == "prior_clean_carry_forward"
    assert artifact["total_arc_games_solved"] == 13

    skipped = artifact["flagged_artifacts_skipped"]
    assert [row["experiment_id"] for row in skipped] == [4146, 4147, 4148, 4149, 4150, 4151, 4153]

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    for experiment_id, row in provenance.items():
        expected_sha = hashlib.sha256(
            mod.DEFAULT_UPSTREAM_PATHS[experiment_id].read_bytes()
        ).hexdigest()
        assert row["sha256"] == expected_sha
    for experiment_id in [4146, 4147, 4148, 4149, 4150, 4151, 4153]:
        assert provenance[experiment_id]["fields_imported"] == []
        assert provenance[experiment_id]["skipped"] is True
    assert "flagged_for_v385" in provenance[4152]["fields_imported"]
    assert "kv260_terminal_confirmed" in provenance[4154]["fields_imported"]


def test_req_capstone_4155_clean_fixture_validates_diffusiongemma(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4155: clean convergence plus value-added graft resolves positive."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "baseline_converged_graft_validated"
    assert artifact["honest_verdict"].startswith(
        "success: capstone_v384_baseline_converged_graft_validated_epochfix1_"
        "baseline0871_graft_ran_value_added"
    )
    assert artifact["baseline_val_trajectory"]["exp4146_epoch_fix_unstalled"] is True
    assert artifact["baseline_val_trajectory"]["matches_published_087"] is True
    assert artifact["baseline_val_trajectory"]["rounded_values"] == pytest.approx(
        [0.106, 0.2782, 0.45, 0.62, 0.78, 0.872]
    )
    assert artifact["graft_verdict"]["status"] == "ran_value_added"
    assert artifact["graft_verdict"]["verifier_value_added"] is True
    assert artifact["diffusiongemma_gate_status"] == "RESOLVED-positive"
    assert artifact["total_arc_games_solved"] == 14


def test_req_capstone_4155_null_deferred_and_accumulating_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4155: null, deferred, and still-climbing headlines are distinct."""

    payloads = _clean_payloads()
    payloads[4150]["verifier_value_added"] = False
    payloads[4150]["rerank_lift_vs_vote"]["delta"] = 0.0
    payloads[4150]["rerank_lift_vs_vote"]["ci95"] = [-0.01, 0.02]
    payloads[4150]["rft_vs_ablation_delta"]["delta"] = 0.0
    payloads[4150]["rft_vs_ablation_delta"]["ci95"] = [-0.02, 0.03]
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.1)
    assert null["headline_outcome"] == "baseline_converged_graft_null"
    assert null["graft_verdict"]["status"] == "ran_null"
    assert null["diffusiongemma_gate_status"] == "RESOLVED-null"

    payloads = _clean_payloads()
    payloads[4150]["graft_deferred"] = True
    payloads[4150]["verifier_value_added"] = False
    payloads[4150]["rft_vs_ablation_delta"]["status"] = "deferred_baseline_below_0.85"
    _write_default_artifacts(tmp_path, payloads)
    deferred = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.1)
    assert deferred["headline_outcome"] == "baseline_converged_graft_deferred"
    assert deferred["graft_verdict"]["status"] == "deferred"
    assert deferred["diffusiongemma_gate_status"] == "STILL-PENDING"

    payloads = _clean_payloads()
    payloads[4149]["matches_published_087"] = False
    payloads[4149]["val_exact_accuracy"] = 0.74
    payloads[4149]["val_trajectory_v384"][-1]["val_exact_accuracy"] = 0.74
    payloads[4149]["val_trajectory_v384"][-1]["effective_val_exact_accuracy"] = 0.74
    payloads[4150]["graft_deferred"] = True
    payloads[4150]["verifier_value_added"] = False
    _write_default_artifacts(tmp_path, payloads)
    accumulating = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.1)
    assert accumulating["headline_outcome"] == "accumulation_unstalled_still_climbing_v385"
    assert accumulating["baseline_val_trajectory"]["final_val_exact_accuracy"] == pytest.approx(0.74)


def test_scenario_capstone_4155_write_and_validate(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4155: write_artifact emits the deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4155_capstone_v384.json"),
        started_s=6.0,
        now_s=6.5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4155
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4155_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4155: validation protects headline and provenance integrity."""

    _write_default_artifacts(tmp_path, _clean_payloads())
    artifact = mod.build_artifact(tmp_path, started_s=7.0, now_s=7.1)

    artifact["headline_outcome"] = "not_enumerated"
    with pytest.raises(ValueError, match="headline_outcome"):
        mod.validate_artifact(artifact)

    artifact["headline_outcome"] = "baseline_converged_graft_validated"
    artifact["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(artifact)

    artifact["upstream_provenance"][0]["sha256"] = "a" * 64
    artifact["baseline_val_trajectory"]["values"] = "bad"
    with pytest.raises(ValueError, match="baseline_val_trajectory"):
        mod.validate_artifact(artifact)

    artifact["baseline_val_trajectory"]["values"] = [0.1]
    artifact["diffusiongemma_gate_status"] = "UNKNOWN"
    with pytest.raises(ValueError, match="diffusiongemma_gate_status"):
        mod.validate_artifact(artifact)

    assert mod.bool_metric({"x": 1}, "x") is None
    assert mod.bool_metric({"x": False}, "x") is False
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": "0.1"}, "x") is None
    assert mod.list_float_metric({"x": ["bad", 0.1]}, "x") == [0.1]
    assert mod.list_float_metric({"x": "bad"}, "x") == []
    assert mod.clean_v384_points("bad") == []
    assert mod.clean_v384_points([42, {"val_exact_accuracy": None}]) == []
    assert mod.baseline_val_trajectory({}, root=tmp_path, skipped_ids=set())["status"] == "missing"
    assert mod.experiment_id_from_name("not_an_experiment") == 0

    fallback_root = tmp_path / "fallback_baseline"
    fallback_root.mkdir()
    assert mod.prior_baseline_values(fallback_root)["values"] == []
    _write_json(
        fallback_root / mod.PRIOR_BASELINE_REFERENCE_PATH,
        {"baseline_val_trajectory": {"values": [0.02317708358168602, 0.105989582837, 0.278172343969]}},
    )
    fallback_payloads = _clean_payloads()
    fallback_payloads[4149].pop("val_trajectory_v384")
    fallback_baseline = mod.baseline_val_trajectory(
        {experiment_id: fallback_payloads[experiment_id] for experiment_id in [4146, 4149]},
        root=fallback_root,
        skipped_ids=set(),
    )
    assert fallback_baseline["rounded_values"] == pytest.approx([0.106, 0.2782, 0.45, 0.872])

    assert mod.graft_verdict_answer({"flagged_adversarial": True}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.arc_games_answer({"flagged_adversarial": True}, was_skipped=True, root=tmp_path)[
        "status"
    ] == "prior_clean_carry_forward"
    assert mod.arc_games_answer(None, was_skipped=False, root=tmp_path)["status"] == (
        "prior_clean_carry_forward"
    )
    no_arc_root = tmp_path / "no_arc"
    no_arc_root.mkdir()
    assert mod.prior_arc_reference(no_arc_root) is None
    assert mod.arc_games_answer(None, was_skipped=False, root=no_arc_root)["status"] == "missing"
    _write_json(no_arc_root / mod.PRIOR_ARC_REFERENCE_PATH, {"total_arc_games_solved": 0})
    assert mod.prior_arc_reference(no_arc_root) is None

    missing_root = tmp_path / "missing_upstream"
    payloads = _clean_payloads()
    payloads.pop(4154)
    _write_default_artifacts(missing_root, payloads)
    missing = mod.build_artifact(missing_root, started_s=8.0, now_s=8.1)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4154}]
    assert {row["experiment_id"] for row in missing["upstream_provenance"]} == set(
        mod.UPSTREAM_IDS
    ) - {4154}


def test_req_capstone_4155_main_prints_written_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """REQ-CAPSTONE-4155: CLI main delegates to write_artifact."""

    output = tmp_path / "results" / "experiment_4155_capstone_v384.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output)

    assert mod.main() == 0
    assert str(output) in capsys.readouterr().out
