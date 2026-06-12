"""Tests for Exp 4075 .376 capstone aggregation.

Spec refs: REQ-CAPSTONE-4075, SCENARIO-CAPSTONE-4075.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v376_4075 as mod


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
        4066: {
            "honest_verdict": "success: archived_v375_v376_active",
            "milestone_375_closestate": {
                "accuracy": {"total_games_solved": 8},
                "g1_off_arc_transfer": {"accumulated_n": 0},
                "g3_decentralization_moe_base": {"accumulated_n": 14},
            },
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4067: {
            "honest_verdict": "complete: sota_ingestion_v376_mapped",
            "methods_mapped_count": 10,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4068: {
            "honest_verdict": "complete: offarc_best_arm_excl0_evalplus_n160",
            "evaluation_corpus": "EvalPlus",
            "corpus": "evalplus",
            "corpus_routed_reason": "12B oracle headroom present on EvalPlus",
            "accumulated_n_tasks": 160,
            "powered_task_floor": 160,
            "oracle_passrate": 0.75,
            "oracle_headroom_present": True,
            "armA_vote_passrate": 0.45,
            "armB_demofit_passrate": 0.465,
            "demofit_delta_pp": 1.5,
            "demofit_bootstrap_ci95": [-0.1, 3.2],
            "demofit_ci_excludes_zero": False,
            "best_arm": "armC_symbolic",
            "best_arm_delta_pp": 4.2,
            "best_arm_ci95": [1.0, 7.5],
            "best_arm_ci_excludes_zero": True,
            "mechanism": mod.MECHANISM,
        },
        4069: {
            "honest_verdict": "complete: decentralization_moe_latent_n30",
            "moe_base_demo_perfect_coverage": 0.43,
            "accumulated_n_tasks": 30,
            "target_n_tasks": 30,
            "coverage_delta_vs_12b": 0.1719,
            "bootstrap_ci95": [0.03, 0.31],
            "oracle_coverage": 0.61,
            "local_support_diagnosis": "latent",
            "mechanism": mod.MECHANISM,
        },
        4070: {
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
        4071: {
            "honest_verdict": "success: verifier_pruner_cuts_actions_equal_solverate",
            "action_reduction_pct": 37.5,
            "solverate_parity_held": True,
            "solverate_baseline": 0.5,
            "solverate_pruned": 0.5,
            "actions_baseline_mean": 160.0,
            "actions_pruned_mean": 100.0,
        },
        4072: {
            "honest_verdict": "complete: arcmemo_v9_cross_game_transfer_win",
            "cross_game_transfer_win": True,
            "actions_cold": 16,
            "actions_within_game": 9,
            "actions_cross_game_v9": 6,
            "induction_calls_cold": 1,
            "induction_calls_within_game": 1,
            "induction_calls_cross_game_v9": 0,
            "n_prior_fragments": 8,
            "n_named_abstractions": 6,
            "n_reused_abstractions": 3,
            "transfer_assessment": "cross_game_v9_win",
        },
        4073: {
            "honest_verdict": "complete: gap4_reeval_bitexact",
            "offline_reeval_bitexact": True,
            "registry_updated": True,
            "gaps_updated": True,
            "g1_off_arc_outcome_recorded": "g1_evalplus_excl0",
            "g3_decentralization_outcome_recorded": "g3_decentralization_latent",
        },
        4074: {
            "honest_verdict": "complete: hardware_continuity",
            "kv260_terminal_confirmed": True,
            "kv260_step_taken": "kv260_opportunistic_ssh_confirm_only",
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "fabric_acceleration_claimed": False,
            "speedup_claim_made": False,
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "per_board_terminal_state": {
                "kv260": "opportunistic_terminal_confirmed_ssh_only",
                "gatemate": "blocked_gatemate_unreachable",
                "polarfire": "reachable_hash_verified_cpu_dispatch_recorded",
            },
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4075_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4075: OpenSpec declares the .376 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4075" in spec
    assert "SCENARIO-CAPSTONE-4075" in spec
    assert "mechanism_fix_produced_measurement" in spec
    assert "off_arc_accumulated_n" in spec
    assert "flagged-skipped" in spec


def test_scenario_capstone_4075_current_artifacts_emit_honest_headline() -> None:
    """SCENARIO-CAPSTONE-4075: landed artifacts skip flagged G1 and preserve clean axes."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses=_summary_statuses(returncodes={4068: 2}),
        started_s=10.0,
        now_s=12.0,
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v376_offarc_flagged_skipped_g3_absent_efficiency_gain_games9"
    )
    assert artifact["mechanism_fix_produced_measurement"] is True
    assert artifact["mechanism_fix_measurement_evidence"]["clean_g3_accumulated_n"] == 30
    assert artifact["mechanism_fix_measurement_evidence"]["clean_off_arc_accumulated_n"] == 0

    assert artifact["verifier_transferred_off_arc_significantly"] is False
    assert artifact["off_arc_accumulated_n"] == 0
    assert artifact["g1_off_arc_transfer"]["status"] == "skipped_flagged"
    assert artifact["g1_off_arc_transfer"]["previous_v375_accumulated_n"] == 0
    assert artifact["g1_off_arc_transfer"]["decision_grade"] is False

    assert artifact["decentralization_diagnosis"] == "absent"
    assert artifact["g3_decentralization_moe_sync"]["accumulated_n"] == 30
    assert artifact["g3_decentralization_moe_sync"]["accumulated_coverage"] == 0.2333
    assert artifact["g3_decentralization_moe_sync"]["baseline_12b_coverage"] == 0.2581
    assert artifact["g3_decentralization_moe_sync"]["decision_grade"] is True

    assert artifact["verifier_pruner_efficiency_gain"] is True
    assert artifact["efficiency_action_pruner"]["action_reduction_pct"] == 66.6667
    assert artifact["efficiency_action_pruner"]["solverate_parity_held"] is True

    assert artifact["total_games_solved"] == 9
    assert artifact["accuracy_self_learning_hardware_deltas"]["accuracy"]["games_solved_delta"] == 1
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["self_learning"][
            "cross_game_transfer_win"
        ]
        is False
    )
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["self_learning"][
            "cross_game_extra_actions_vs_within_game"
        ]
        == 1
    )
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["hardware"]["gatemate_step_taken"]
        == "blocked_gatemate_unreachable"
    )
    assert artifact["verifier_registry_and_gaps_hygiene"]["gaps_updated"] is True

    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4068]
    assert artifact["missing_upstream_artifacts"] == []
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == {4066, 4067, 4069, 4070, 4071, 4072, 4073, 4074}
    assert cited[4071] == {
        "experiment_id": 4071,
        "sha256": hashlib.sha256(
            Path("results/experiment_4071_verifier_action_pruner_efficiency.json").read_bytes()
        ).hexdigest(),
    }
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["mechanism_fix_produced_measurement"].startswith(
        "BARE BOOL"
    )


def test_req_capstone_4075_clean_fixture_records_positive_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4075: clean upstreams can satisfy mechanism, G1, G3, and efficiency."""

    payloads = _fixture_payloads()
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path, summary_statuses=_summary_statuses(), started_s=1.0, now_s=1.25
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v376_offarc_measured_n160_g3_latent_efficiency_gain_games9"
    )
    assert artifact["mechanism_fix_produced_measurement"] is True
    assert artifact["verifier_transferred_off_arc_significantly"] is True
    assert artifact["off_arc_accumulated_n"] == 160
    assert artifact["g1_off_arc_transfer"]["decision_grade"] is True
    assert artifact["g1_off_arc_transfer"]["excludes_zero_with_headroom"] is True
    assert artifact["decentralization_diagnosis"] == "latent"
    assert artifact["g3_decentralization_moe_sync"]["decision_grade"] is True
    assert artifact["verifier_pruner_efficiency_gain"] is True
    assert artifact["total_games_solved"] == 9
    assert (
        artifact["accuracy_self_learning_hardware_deltas"]["self_learning"][
            "action_savings_vs_cold"
        ]
        == 10
    )
    assert artifact["flagged_artifacts_skipped"] == []
    assert len(artifact["cited_upstream_artifacts"]) == len(payloads)
    assert artifact["duration_s"] == 0.25


def test_req_capstone_4075_flagged_and_missing_inputs_cannot_leak_metrics(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4075: skipped or missing upstreams do not satisfy required axes."""

    payloads = _fixture_payloads()
    payloads[4068]["flagged_adversarial"] = True
    payloads[4069]["flagged_adversarial"] = True
    payloads[4071]["flagged_adversarial"] = True
    payloads.pop(4070)
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(
            tuple(payloads), returncodes={4068: 2, 4069: 2, 4071: 2}
        ),
    )

    mod.validate_artifact(artifact)
    assert artifact["mechanism_fix_produced_measurement"] is False
    assert artifact["verifier_transferred_off_arc_significantly"] is False
    assert artifact["off_arc_accumulated_n"] == 0
    assert artifact["g1_off_arc_transfer"]["status"] == "skipped_flagged"
    assert artifact["decentralization_diagnosis"] == "flagged_skipped"
    assert artifact["g3_decentralization_moe_sync"]["accumulated_n"] == 0
    assert artifact["verifier_pruner_efficiency_gain"] is False
    assert artifact["efficiency_action_pruner"]["status"] == "skipped_flagged"
    assert artifact["total_games_solved"] == 8
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 4070}]
    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [
        4068,
        4069,
        4071,
    ]
    assert {4068, 4069, 4071}.isdisjoint(
        {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    )


def test_req_capstone_4075_write_artifact_and_validate_rejects_regressions(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4075: artifact writing validates required schema fields."""

    _write_default_artifacts(tmp_path, _fixture_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.5,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert written["schema"] == "carnot.capstone_v376_4075.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["mechanism_fix_produced_measurement"] = "true"
    with pytest.raises(ValueError, match="bare bool"):
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
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4069, "sha256": "not-sha"}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["flagged_artifacts_skipped"] = {}
    with pytest.raises(ValueError, match="list"):
        mod.validate_artifact(bad)


def test_req_capstone_4075_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4075: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / mod.DEFAULT_UPSTREAM_PATHS[4069]
    _write_json(path, _fixture_payloads()[4069])
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

    statuses = mod.summarize_existing_artifacts(tmp_path, {4069: path, 4071: None}, None)
    assert statuses == {4069: {"returncode": 0, "stdout": "summary", "stderr": ""}}
    assert mod.summarize_existing_artifacts(tmp_path, {4069: path}, {4069: {"returncode": 2}}) == {
        4069: {"returncode": 2}
    }

    fallback = tmp_path / "results" / "experiment_4069_alt.json"
    _write_json(fallback, _fixture_payloads()[4069])
    selected = mod.selected_upstream_paths(tmp_path)
    assert selected[4069] == path

    path.unlink()
    selected = mod.selected_upstream_paths(tmp_path)
    assert selected[4069] == fallback

    assert mod.g1_off_arc_report(None, was_skipped=False)["status"] == "missing_or_blocked"
    assert (
        mod.g1_off_arc_report(
            {
                "honest_verdict": "complete: offarc_accumulating",
                "accumulated_n_tasks": 10,
                "powered_task_floor": 160,
                "mechanism": mod.MECHANISM,
            },
            was_skipped=False,
        )["status"]
        == "accumulating_n_10"
    )
    assert (
        mod.g1_off_arc_report(
            {
                "honest_verdict": "complete: no_headroom",
                "accumulated_n_tasks": 160,
                "powered_task_floor": 160,
                "mechanism": mod.MECHANISM,
            },
            was_skipped=False,
        )["status"]
        == "uninformative_no_oracle_headroom"
    )
    assert (
        mod.g1_off_arc_report(
            {
                "honest_verdict": "complete: measured_not_sig",
                "accumulated_n_tasks": 160,
                "powered_task_floor": 160,
                "oracle_headroom_present": True,
                "best_arm_delta_pp": 1.0,
                "best_arm_ci_excludes_zero": False,
                "mechanism": mod.MECHANISM,
            },
            was_skipped=False,
        )["status"]
        == "measured_not_significant"
    )
    assert (
        mod.g1_off_arc_report(
            {
                "honest_verdict": "complete: empty",
                "accumulated_n_tasks": 0,
                "powered_task_floor": 160,
            },
            was_skipped=False,
        )["status"]
        == "accumulating_n_0"
    )
    assert mod.g3_decentralization_report({}, was_skipped=False)["status"] == "missing_or_blocked"
    assert (
        mod.g3_decentralization_report(
            {
                "honest_verdict": "complete: uninformative",
                "accumulated_n_tasks": 30,
                "target_n_tasks": 30,
                "local_support_diagnosis": "uninformative",
            },
            was_skipped=False,
        )["diagnosis"]
        == "uninformative"
    )
    assert (
        mod.g3_decentralization_report(
            {
                "honest_verdict": "complete: accumulating",
                "accumulated_n_tasks": 10,
                "target_n_tasks": 30,
                "local_support_diagnosis": "latent",
            },
            was_skipped=False,
        )["diagnosis"]
        == "accumulating"
    )
    assert (
        mod.g3_decentralization_report(
            {
                "honest_verdict": "complete: derived",
                "accumulated_n_tasks": 30,
                "target_n_tasks": 30,
                "moe_base_demo_perfect_coverage": 0.30,
            },
            was_skipped=False,
        )["diagnosis"]
        == "latent"
    )
    assert mod.efficiency_report(None, was_skipped=True)["status"] == "skipped_flagged"
    assert mod.nested_int({"a": []}, ("a", "b")) == 0
    assert "offarc_accumulating_n10" in mod.verdict(
        g1={"status": "accumulating_n_10", "accumulated_n_tasks": 10, "powered_task_floor": 160},
        g3={"diagnosis": "accumulating"},
        efficiency_gain=False,
        total_games_solved=8,
        skipped_count=0,
    )
    assert "offarc_accumulating_n0" in mod.verdict(
        g1={"status": "accumulating_n_0", "accumulated_n_tasks": 0, "powered_task_floor": 160},
        g3={"diagnosis": "accumulating"},
        efficiency_gain=False,
        total_games_solved=8,
        skipped_count=0,
    )
