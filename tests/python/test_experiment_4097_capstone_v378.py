"""Tests for Exp 4097 .378 capstone aggregation.

Spec refs: REQ-CAPSTONE-4097, SCENARIO-CAPSTONE-4097.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v378_4097 as mod


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


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4086: {
            "honest_verdict": "success: archived_v377_v378_active",
            "milestone_377_closestate": {
                "sudoku_control": {"reproduces_beachhead": False},
                "accuracy": {"total_games_solved": 9},
            },
            "total_games_solved": 9,
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4087: {
            "honest_verdict": "complete: precision_rescue_succeeded_best_0.9000_at_recall_0.3000",
            "precision_rescue_succeeded": True,
            "best_certified_precision": 0.9,
            "best_op_point_recall": 0.3,
            "best_operating_point": {"filter_stack": "fixture", "threshold": "k=2"},
        },
        4088: {
            "honest_verdict": "complete: rft_corpus_built_3arms",
            "runner_ready": True,
            "trainer_smoke_passed": True,
            "n_rft_correct": 12,
            "n_rft_ablation": 12,
            "n_gold_sft": 12,
        },
        4089: {
            "honest_verdict": "complete: verifier_reward_rft_train_finished",
            "train_launched": True,
            "training_accumulating": False,
            "epochs_completed": {"rft_correct": 3, "rft_ablation": 3, "gold_sft": 3},
        },
        4090: {
            "honest_verdict": "success: rft_correct_A_beats_ablation_B",
            "status": "complete",
            "a_vs_b_delta": 0.12,
            "a_vs_b_ci95": [0.03, 0.2],
            "rft_correct_pass_at_1": 0.44,
            "rft_ablation_pass_at_1": 0.32,
            "n_heldout_tasks": 36,
        },
        4091: {
            "honest_verdict": "complete: sudoku_pipeline_sanity_reproduced",
            "sudoku_sanity_reproduced": True,
            "reproduces_beachhead": True,
        },
        4092: {
            "honest_verdict": "success: tenth_game_solved_fixture_at_action_4",
            "game_solved": True,
            "real_env_confirmed": True,
            "prior_total_games_solved": 9,
            "total_games_solved": 10,
            "target_game": "fixture-game",
        },
        4093: {
            "honest_verdict": "complete: offarc_demofit_precision_0.80_filter_raises_to_0.88",
            "demofit_precision_raw": 0.8,
            "demofit_precision_filtered": 0.88,
            "filter_recall": 0.7,
            "primitive_is_domain_general": True,
            "domain_general_precision_floor": 0.68,
            "n_tasks_scored": 160,
        },
        4094: {
            "honest_verdict": "complete: sota_ingestion_precision_calibration_mapped",
            "methods_mapped": [{"arxiv_id": "2411.02272", "one_line": "fixture"}],
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4095: {
            "honest_verdict": "complete: verifier_registry_gaps_hygiene",
            "registry_updated": True,
            "gaps_updated": True,
        },
        4096: {
            "honest_verdict": "complete: hardware_continuity",
            "kv260_terminal_confirmed": True,
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4097_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4097: OpenSpec declares the .378 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4097" in spec
    assert "SCENARIO-CAPSTONE-4097" in spec
    assert "precision_rescue_outcome" in spec
    assert "offarc_precision_transfer" in spec
    assert "sudoku_sanity_reproduced" in spec


def test_scenario_capstone_4097_current_artifacts_report_honest_truth() -> None:
    """SCENARIO-CAPSTONE-4097: current artifacts skip flagged registry evidence."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses=_summary_statuses(returncodes={4095: 2}),
        started_s=10.0,
        now_s=12.0,
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v378_precision_rescued_0.8824_"
        "phaseB_no_clean_A_vs_B_games10_offarc_transfer_flagged_skipped1"
    )
    assert artifact["precision_rescue"]["rescued"] is True
    assert artifact["precision_rescue"]["best_certified_precision"] == 0.8824
    assert artifact["precision_rescue"]["best_op_point_recall"] == 0.7143
    assert artifact["precision_rescue_outcome"] == (
        "principle: THE gate passed; certification precision moved from the 0.68 "
        "baseline to 0.8824 at recall 0.7143."
    )
    assert artifact["pivot_comparison"]["status"] == "missing"
    assert artifact["pivot_result"] == (
        "principle: precision rescue passed, but no clean exp4090 A-vs-B artifact "
        "exists; exp4088/exp4089 stopped before a decision-grade held-out RFT eval, "
        "so verifier-label training signal is unmeasured."
    )
    assert artifact["offarc_precision"]["primitive_is_domain_general"] is True
    assert artifact["offarc_precision_transfer"].startswith(
        "principle: precision primitive transfers off-ARC"
    )
    assert artifact["games_solved_total"] == 10
    assert artifact["arc_accuracy"]["game_solved"] is True
    assert artifact["sudoku_sanity_reproduced"] is False
    assert artifact["sudoku_sanity"]["status"] == "missing"
    assert "gated_on" not in artifact

    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4095]
    assert [row["experiment_id"] for row in artifact["missing_upstream_artifacts"]] == [4090, 4091]
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == {4086, 4087, 4088, 4089, 4092, 4093, 4094, 4096}
    assert cited[4092] == {
        "experiment_id": 4092,
        "sha256": hashlib.sha256(
            Path("results/experiment_4092_tenth_game_explore_first.json").read_bytes()
        ).hexdigest(),
    }
    assert artifact["upstream_artifact_state"]["4095"]["skipped"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_req_capstone_4097_clean_fixture_records_rft_a_gt_b(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4097: clean exp4090 evidence can satisfy the RFT A-vs-B gate."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(
        tmp_path, summary_statuses=_summary_statuses(), started_s=1.0, now_s=1.5
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith(
        "success: capstone_v378_precision_rescued_0.9000_"
        "rft_A_gt_B_games10_offarc_transfer_sudoku_reproduced"
    )
    assert artifact["pivot_comparison"]["status"] == "a_gt_b"
    assert artifact["pivot_comparison"]["a_vs_b_ci_excludes_zero"] is True
    assert artifact["pivot_result"] == (
        "principle: clean exp4090 measured A>B with CI excluding zero; the "
        "verifier label carries training signal."
    )
    assert artifact["sudoku_sanity_reproduced"] is True
    assert artifact["offarc_precision_transfer"] == (
        "principle: precision primitive transfers off-ARC; raw demo-fit precision "
        "0.8000 and filtered precision 0.8800 clear the 0.6800 floor."
    )
    assert artifact["flagged_artifacts_skipped"] == []
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )


def test_req_capstone_4097_failed_rescue_reports_bounded_forward_path(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4097: failed precision rescue honestly bounds verifier-as-reward."""

    payloads = _clean_payloads()
    for experiment_id in (4088, 4089, 4090, 4091):
        payloads.pop(experiment_id)
    payloads[4087] = {
        "honest_verdict": "complete: precision_rescue_FAILED_max_0.7400",
        "precision_rescue_succeeded": False,
        "best_certified_precision": 0.74,
        "best_op_point_recall": 0.5,
    }
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path, summary_statuses=_summary_statuses(), started_s=2.0, now_s=2.25
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v378_precision_bounded_max_0.7400_"
        "phaseB_skipped_honest_bound_games10"
    )
    assert artifact["precision_rescue"]["rescued"] is False
    assert artifact["precision_rescue_outcome"] == (
        "principle: THE gate failed; max certification precision was 0.7400 at recall "
        "0.5000, below the 0.8500 precision floor."
    )
    assert artifact["pivot_comparison"]["status"] == "skipped_bounded"
    assert artifact["pivot_result"] == (
        "principle: verifier-as-reward is precision-bounded on ARC; Phase B is skipped "
        "honestly, and the forward path is step-level process-reward / outcome-verifier pairing."
    )
    assert artifact["missing_upstream_artifacts"] == [
        {"experiment_id": 4088},
        {"experiment_id": 4089},
        {"experiment_id": 4090},
        {"experiment_id": 4091},
    ]


def test_scenario_capstone_4097_write_artifact_validates_output(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4097: write_artifact emits the required deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4097_capstone_v378.json"),
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.25,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4097
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4097_branch_states_are_explicit(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4097: missing, accumulating, null, and validation states stay explicit."""

    fallback = tmp_path / "results/experiment_4090_fixture.json"
    _write_json(fallback, {"honest_verdict": "complete: fallback"})
    paths = mod.selected_upstream_paths(tmp_path)
    statuses = mod.summarize_existing_artifacts(tmp_path, paths, {4090: {"returncode": 0}})
    assert paths[4090] == fallback
    assert paths[4086] is None
    assert statuses == {4090: {"returncode": 0}}

    assert mod.float_metric({"x": True}, "x") == 0.0
    assert mod.int_metric({"x": False}, "x") == 0
    assert mod.list_float_metric({"ci": "bad"}, "ci") == []
    assert mod.nested_bool({"outer": {"inner": True}}, ("outer", "inner")) is True
    assert mod.nested_bool(None, ("outer",)) is False
    assert mod.nested_int({"outer": {"inner": 3}}, ("outer", "inner")) == 3
    assert mod.nested_int({"outer": {"inner": True}}, ("outer", "inner")) == 0
    assert mod.is_sha256("0" * 64) is True
    assert mod.is_sha256("not-sha") is False

    skipped_rescue = mod.precision_rescue({"flagged_adversarial": True}, was_skipped=True)
    assert skipped_rescue["status"] == "skipped_flagged"
    blocked_rescue = mod.precision_rescue({"honest_verdict": "blocked_cache_missing"}, was_skipped=False)
    assert blocked_rescue["status"] == "blocked"

    assert mod.pivot_comparison(None, precision_rescued=False, was_skipped=False)["status"] == (
        "skipped_bounded"
    )
    skipped_pivot = mod.pivot_comparison(None, precision_rescued=True, was_skipped=True)
    assert skipped_pivot["status"] == "skipped_flagged"
    assert mod.pivot_result_text(skipped_pivot) == (
        "principle: exp4090 was flagged_adversarial and skipped; A-vs-B is unmeasured."
    )
    assert mod.pivot_comparison(None, precision_rescued=True, was_skipped=False)["status"] == (
        "missing"
    )
    blocked_pivot = mod.pivot_comparison(
        {"honest_verdict": "blocked_gate_check_failed"},
        precision_rescued=True,
        was_skipped=False,
    )
    assert blocked_pivot["status"] == "blocked"
    assert mod.pivot_result_text(blocked_pivot).startswith(
        "principle: precision rescue passed, but Phase B blocked"
    )
    accumulating = mod.pivot_comparison(
        {"honest_verdict": "complete: train accumulating", "training_accumulating": True},
        precision_rescued=True,
        was_skipped=False,
    )
    assert accumulating["status"] == "accumulating"
    assert mod.pivot_result_text(accumulating).startswith(
        "principle: Phase B is still accumulating"
    )
    null = mod.pivot_comparison(
        {"honest_verdict": "complete: null", "a_vs_b_delta": 0.01, "a_vs_b_ci95": [-0.02, 0.04]},
        precision_rescued=True,
        was_skipped=False,
    )
    assert null["status"] == "measured_null"
    assert mod.pivot_result_text(null).startswith("principle: clean exp4090 did not measure A>B")
    no_numbers = mod.pivot_comparison(
        {"honest_verdict": "complete: null_no_numbers"},
        precision_rescued=True,
        was_skipped=False,
    )
    assert no_numbers["a_vs_b_delta"] == 0.0
    assert no_numbers["a_vs_b_ci95"] == []
    assert no_numbers["status"] == "measured_null"

    assert mod.offarc_precision(None, was_skipped=True)["status"] == "skipped_flagged"
    assert mod.offarc_precision({"honest_verdict": "blocked_cache_missing"}, was_skipped=False)[
        "status"
    ] == "blocked"
    bounded_offarc = mod.offarc_precision(
        {
            "honest_verdict": "complete: offarc_bounded",
            "primitive_is_domain_general": False,
            "demofit_precision_raw": 0.61,
            "demofit_precision_filtered": 0.62,
        },
        was_skipped=False,
    )
    assert bounded_offarc["status"] == "bounded"
    assert mod.offarc_precision_transfer_text(bounded_offarc).startswith(
        "principle: off-ARC precision replay landed"
    )

    assert mod.sudoku_sanity(None, None, was_skipped=True)["status"] == "skipped_flagged"
    assert mod.sudoku_sanity({"honest_verdict": "blocked_train_missing"}, None, was_skipped=False)[
        "status"
    ] == "blocked"
    assert mod.verdict(
        rescue={"rescued": True, "best_certified_precision": 0.9},
        pivot=accumulating,
        offarc={"primitive_is_domain_general": False},
        sudoku={"sudoku_sanity_reproduced": False},
        games_solved_total=10,
        skipped_count=0,
    ).startswith("complete: capstone_v378_precision_rescued_0.9000_phaseB_accumulating")

    bad = mod.build_artifact(
        tmp_path, summary_statuses={4090: {"returncode": 0}}, started_s=1.0, now_s=1.1
    )
    bad["games_solved_total"] = True
    with pytest.raises(ValueError, match="games_solved_total"):
        mod.validate_artifact(bad)
