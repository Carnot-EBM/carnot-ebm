"""Tests for Exp 4065 .375 capstone aggregation.

Spec refs: REQ-CAPSTONE-4065, SCENARIO-CAPSTONE-4065.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v375_4065 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_statuses(
    experiment_ids: tuple[int, ...] = mod.UPSTREAM_IDS,
    *,
    returncodes: dict[int, int] | None = None,
) -> dict[int, JsonDict]:
    overrides = returncodes or {}
    return {
        experiment_id: {
            "returncode": overrides.get(experiment_id, 0),
            "stdout": f"summarized {experiment_id}",
            "stderr": "",
        }
        for experiment_id in experiment_ids
    }


def _fixture_payloads() -> dict[int, JsonDict]:
    return {
        4054: {
            "honest_verdict": "success: archived_v374_v375_active",
            "milestone_374_closestate": {
                "accuracy": {"total_games_solved": 8},
                "g3_decentralization_moe_base": {
                    "baseline_12b_coverage": 0.2581,
                    "checkpoint_n_tasks": 14,
                    "moe_base_coverage": 0.3571,
                    "bootstrap_ci95": [0.143, 0.643],
                    "target_task_floor": 30,
                    "operator_corrected_diagnosis": "underpowered_not_retired",
                },
            },
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4055: {
            "honest_verdict": "complete: sota_ingestion_unsaturated_execverif_and_pruner_mapped",
            "methods_mapped_count": 10,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4056: {
            "honest_verdict": "success: runner_built",
            "runner_ready": True,
            "flagged_adversarial": True,
        },
        4057: {
            "honest_verdict": "complete: offarc_power_evalplus_full_n_best_arm_excl0",
            "accumulated_n_tasks": 160,
            "powered_task_floor": 160,
            "best_arm": "armC_symbolic",
            "best_arm_delta_pp": 4.2,
            "best_arm_ci95": [1.0, 7.5],
            "best_arm_ci_excludes_zero": True,
            "demofit_delta_pp": 1.5,
            "demofit_bootstrap_ci95": [-0.1, 3.2],
            "demofit_ci_excludes_zero": False,
            "oracle_passrate": 0.75,
            "oracle_headroom_present": True,
            "raw_artifact_present": True,
            "partial_reason": "",
        },
        4059: {
            "honest_verdict": "complete: decentralization_moe_resume_latent_n30",
            "accumulated_n": 30,
            "target_task_floor": 30,
            "accumulated_coverage": 0.43,
            "coverage_delta_vs_12b": 0.1719,
            "bootstrap_ci95": [0.29, 0.58],
            "local_support_diagnosis": "latent",
            "raw_complete": True,
        },
        4060: {
            "honest_verdict": "success: ninth_game_solved_at_action_8",
            "prior_total_games_solved": 8,
            "total_games_solved": 9,
            "game_solved": True,
            "real_env_confirmed": True,
            "target_game": "zz99-1234abcd",
            "candidate_baseline_actions": 16,
            "first_solve_at_action": 8,
            "exploration_actions_used": 1,
        },
        4061: {
            "honest_verdict": "complete: verifier_action_pruner_efficiency_gain",
            "action_reduction_pct": 37.5,
            "solverate_parity_held": True,
            "baseline_solverate": 0.5,
            "pruned_solverate": 0.5,
            "baseline_actions": 160,
            "pruned_actions": 100,
        },
        4062: {
            "honest_verdict": "complete: arcmemo_v8_cross_game_transfer_win",
            "cross_game_transfer_win": True,
            "actions_cold": 16,
            "actions_within_game": 9,
            "actions_cross_game_v8": 6,
            "induction_calls_cold": 1,
            "induction_calls_within_game": 1,
            "induction_calls_cross_game_v8": 0,
            "n_prior_fragments": 8,
            "n_named_abstractions": 2,
            "n_reused_abstractions": 1,
            "transfer_assessment": "cross_game_v8_win",
        },
        4063: {
            "honest_verdict": "complete: gap4_reeval_bitexact",
            "offline_reeval_bitexact": True,
            "registry_updated": True,
            "gaps_updated": True,
            "g1_off_arc_outcome_recorded": "g1_evalplus_excl0",
            "g3_decentralization_outcome_recorded": "g3_decentralization_latent",
        },
        4064: {
            "honest_verdict": "complete: hardware_continuity",
            "kv260_terminal_confirmed": True,
            "gatemate_step_taken": "gatemate_existing_n16_bitstream_post_flash_detect_blocked",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "fabric_acceleration_claimed": False,
            "speedup_claim_made": False,
            "per_board_reachability": {"kv260": True, "gatemate": True, "polarfire": True},
            "per_board_terminal_state": {
                "kv260": "opportunistic_terminal_confirmed_ssh_only",
                "gatemate": "reachable_n16_bitstream_post_flash_detect_blocked",
                "polarfire": "reachable_hash_verified_cpu_dispatch_recorded",
            },
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4065_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4065: OpenSpec declares the .375 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4065" in spec
    assert "SCENARIO-CAPSTONE-4065" in spec
    assert "off_arc_accumulated_n" in spec
    assert "verifier_pruner_efficiency_gain" in spec
    assert "accumulating" in spec


def test_scenario_capstone_4065_current_artifacts_emit_honest_headline() -> None:
    """SCENARIO-CAPSTONE-4065: landed artifacts preserve accumulating and missing states."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses=_summary_statuses(returncodes={4056: 2}),
        started_s=10.0,
        now_s=12.0,
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v375_offarc_accumulating_n0_g3_accumulating_efficiency_null_games8"
    )
    assert artifact["verifier_transferred_off_arc_significantly"] is False
    assert artifact["off_arc_accumulated_n"] == 0
    assert artifact["g1_off_arc_evalplus"]["best_arm"] == "armC_symbolic"
    assert artifact["g1_off_arc_evalplus"]["best_arm_ci95"] == [0.0, 0.0]
    assert artifact["g1_off_arc_evalplus"]["best_arm_ci_excludes_zero"] is False
    assert artifact["g1_off_arc_evalplus"]["oracle_headroom_present"] is False
    assert artifact["g1_off_arc_evalplus"]["decision_grade"] is False

    assert artifact["decentralization_diagnosis"] == "accumulating"
    assert artifact["g3_decentralization_moe_resume"]["accumulated_n"] == 14
    assert artifact["g3_decentralization_moe_resume"]["accumulated_coverage"] == 0.3571
    assert artifact["g3_decentralization_moe_resume"]["baseline_12b_coverage"] == 0.2581
    assert artifact["g3_decentralization_moe_resume"]["decision_grade"] is False
    assert artifact["g3_decentralization_moe_resume"]["source"] == "exp4054_resume_checkpoint"

    assert artifact["verifier_pruner_efficiency_gain"] is False
    assert artifact["efficiency_action_pruner"]["status"] == "missing_or_blocked"
    assert artifact["efficiency_action_pruner"]["action_reduction_pct"] == 0.0
    assert artifact["efficiency_action_pruner"]["solverate_parity_held"] is False

    assert artifact["total_games_solved"] == 8
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["status"]
        == "missing_or_blocked"
    )
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["games_solved_delta"] == 0
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["self_learning"][
            "cross_game_transfer_win"
        ]
        is False
    )
    assert artifact["accuracy_self_learning_hardware_deltas"]["self_learning"][
        "transfer_assessment"
    ] == ("unmeasured_no_usable_trace")
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["hardware"]["kv260_terminal_confirmed"]
        is True
    )
    assert artifact["accuracy_self_learning_hardware_deltas"]["hardware"][
        "polarfire_step_taken"
    ] == ("polarfire_hash_verified_cpu_dispatch_succeeded")

    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4056]
    assert artifact["missing_upstream_artifacts"] == [
        {"experiment_id": 4058},
        {"experiment_id": 4059},
        {"experiment_id": 4060},
        {"experiment_id": 4061},
    ]
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == {4054, 4055, 4057, 4062, 4063, 4064}
    assert cited[4057] == {
        "experiment_id": 4057,
        "sha256": hashlib.sha256(
            Path("results/experiment_4057_offarc_power_evalplus.json").read_bytes()
        ).hexdigest(),
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["off_arc_accumulated_n"].startswith("BARE INT")


def test_req_capstone_4065_clean_fixture_can_record_positive_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4065: clean upstream metrics can satisfy off-ARC, G3, and efficiency."""

    payloads = _fixture_payloads()
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path, summary_statuses=_summary_statuses(), started_s=1.0, now_s=1.25
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v375_offarc_excl0_g3_latent_efficiency_gain_games9"
    )
    assert artifact["verifier_transferred_off_arc_significantly"] is True
    assert artifact["off_arc_accumulated_n"] == 160
    assert artifact["g1_off_arc_evalplus"]["decision_grade"] is True
    assert artifact["decentralization_diagnosis"] == "latent"
    assert artifact["g3_decentralization_moe_resume"]["decision_grade"] is True
    assert artifact["verifier_pruner_efficiency_gain"] is True
    assert artifact["efficiency_action_pruner"]["action_reduction_pct"] == 37.5
    assert artifact["efficiency_action_pruner"]["solverate_parity_held"] is True
    assert artifact["total_games_solved"] == 9
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["games_solved_delta"] == 1
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["self_learning"][
            "action_savings_vs_cold"
        ]
        == 10
    )
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 4056,
            "path": "results/experiment_4056_offarc_power_evalplus_build.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    assert len(artifact["cited_upstream_artifacts"]) == len(payloads) - 1
    assert artifact["duration_s"] == 0.25


def test_req_capstone_4065_flagged_and_missing_inputs_cannot_leak_metrics(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4065: skipped or missing upstreams do not satisfy required axes."""

    payloads = _fixture_payloads()
    payloads[4057]["flagged_adversarial"] = True
    payloads[4059]["flagged_adversarial"] = True
    payloads[4061]["flagged_adversarial"] = True
    payloads.pop(4060)
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(
            tuple(payloads), returncodes={4057: 2, 4059: 2, 4061: 2}
        ),
    )

    mod.validate_artifact(artifact)
    assert artifact["verifier_transferred_off_arc_significantly"] is False
    assert artifact["off_arc_accumulated_n"] == 0
    assert artifact["g1_off_arc_evalplus"]["status"] == "skipped_flagged"
    assert artifact["decentralization_diagnosis"] == "flagged_skipped"
    assert artifact["g3_decentralization_moe_resume"]["accumulated_n"] == 0
    assert artifact["verifier_pruner_efficiency_gain"] is False
    assert artifact["efficiency_action_pruner"]["status"] == "skipped_flagged"
    assert artifact["total_games_solved"] == 8
    assert artifact["missing_upstream_artifacts"] == [
        {"experiment_id": 4058},
        {"experiment_id": 4060},
    ]
    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [
        4056,
        4057,
        4059,
        4061,
    ]
    assert {4056, 4057, 4059, 4061}.isdisjoint(
        {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    )


def test_req_capstone_4065_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4065: artifact writing validates required schema fields."""

    _write_default_artifacts(tmp_path, _fixture_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.5,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert written["schema"] == "carnot.capstone_v375_4065.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["verifier_transferred_off_arc_significantly"] = "true"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["off_arc_accumulated_n"] = True
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["decentralization_diagnosis"] = "retired"
    with pytest.raises(ValueError, match="decentralization_diagnosis"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["verifier_pruner_efficiency_gain"] = 1
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["total_games_solved"] = "9"
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="aggregation_from_upstream_artifacts"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4057, "sha256": "not-sha"}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["flagged_artifacts_skipped"] = {}
    with pytest.raises(ValueError, match="list"):
        mod.validate_artifact(bad)


def test_req_capstone_4065_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4065: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / mod.DEFAULT_UPSTREAM_PATHS[4057]
    _write_json(path, _fixture_payloads()[4057])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, 0, stdout="summary", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod.run_summarize_artifact(tmp_path, path) == {
        "returncode": 0,
        "stdout": "summary",
        "stderr": "",
    }
    assert calls == [[str(mod.PYTHON_BIN), "scripts/summarize_artifact.py", str(path)]]

    statuses = mod.summarize_existing_artifacts(tmp_path, {4057: path, 4061: None}, supplied=None)
    assert statuses == {4057: {"returncode": 0, "stdout": "summary", "stderr": ""}}
    assert mod.summarize_existing_artifacts(
        tmp_path, {4057: path}, supplied={4057: {"returncode": 2}}
    ) == {4057: {"returncode": 2}}
    assert mod.list_float_metric({"ci": "not-a-list"}, "ci") == []
    assert mod.list_float_metric({"ci": [1, True, 2.5, "bad"]}, "ci") == [1.0, 2.5]
    assert mod.nested_mapping({"a": {"b": 3}}, ("a",)) == {"b": 3}
    assert mod.nested_int({"a": []}, ("a", "b")) == 0
    assert mod.g1_off_arc_report(None, was_skipped=False)["status"] == "missing_or_blocked"
    assert (
        mod.g1_off_arc_report(
            {
                "honest_verdict": "complete: no_headroom",
                "accumulated_n_tasks": 160,
                "powered_task_floor": 160,
            },
            was_skipped=False,
        )["status"]
        == "uninformative_or_not_significant"
    )
    assert (
        mod.g3_decentralization_report({}, None, was_skipped=False)["status"]
        == "missing_or_blocked"
    )
    assert mod.efficiency_report(None, was_skipped=True)["status"] == "skipped_flagged"
