"""Tests for Exp 4085 .377 capstone aggregation.

Spec refs: REQ-CAPSTONE-4085, SCENARIO-CAPSTONE-4085.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v377_4085 as mod


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
        4076: {
            "honest_verdict": "success: archived_v376_v377_active",
            "milestone_376_closestate": {"accuracy": {"total_games_solved": 9}},
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4077: {
            "honest_verdict": "complete: verifier_reward_corpus_ready",
            "runner_ready": True,
            "trainer_smoke_passed": True,
            "n_rft_correct": 48,
            "n_gold_sft": 48,
        },
        4078: {
            "honest_verdict": "complete: verifier_reward_rft_training_finished",
            "train_launched": True,
            "training_accumulating": False,
            "epochs_completed": {"qwen35_08b:rft_correct": 1, "qwen35_08b:gold_sft": 1},
        },
        4079: {
            "honest_verdict": "success: verifier_reward_rft_beats_cold_matches_gold",
            "status": "complete",
            "base_model": "qwen35_08b",
            "rft_correct_induction_rate": 0.42,
            "cold_base_induction_rate": 0.25,
            "gold_sft_induction_rate": 0.41,
            "rft_vs_cold_delta": 0.17,
            "rft_vs_cold_ci95": [0.04, 0.3],
            "rft_vs_gold_delta": 0.01,
            "rft_vs_gold_ci95": [-0.03, 0.05],
            "codex_induction_rate": 0.57,
            "prior_local_induction_rate": 0.26,
            "training_accumulating": False,
            "inference_substrate": "heldout_arc_induction_eval",
        },
        4080: {
            "honest_verdict": "complete: sudoku_positive_control_rft_ge_sft_reproduced",
            "reproduces_beachhead": True,
            "rft_rate": 0.065833,
            "sft_rate": 0.0505,
            "n_seeds": 3,
        },
        4081: {
            "honest_verdict": "complete: sota_ingestion_verifier_as_reward_mapped",
            "methods_mapped": [{"arxiv_id": "2601.17223"}],
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
        },
        4082: {
            "honest_verdict": "success: tenth_game_solved",
            "game_solved": True,
            "real_env_confirmed": True,
            "total_games_solved": 10,
            "prior_total_games_solved": 9,
            "target_game": "zz99-fixture",
        },
        4083: {
            "honest_verdict": "complete: registry_clean",
            "registry_updated": True,
            "gaps_updated": True,
            "pivot_outcome_recorded": True,
        },
        4084: {
            "honest_verdict": "complete: hardware_continuity",
            "kv260_terminal_confirmed": True,
            "gatemate_step_taken": "gatemate_fixture",
            "polarfire_step_taken": "polarfire_fixture",
            "per_board_reachability": {"kv260": True, "gatemate": True, "polarfire": True},
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4085_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4085: OpenSpec declares the .377 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4085" in spec
    assert "SCENARIO-CAPSTONE-4085" in spec
    assert "sudoku_control_reproduced" in spec
    assert "decentralization_distillation_outcome" in spec
    assert "games_solved_total" in spec


def test_scenario_capstone_4085_current_artifacts_skip_flagged_and_report_blocked_pivot() -> None:
    """SCENARIO-CAPSTONE-4085: current artifacts are aggregated ungated with flagged skips."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses=_summary_statuses(returncodes={4077: 2, 4078: 2, 4080: 2, 4083: 2}),
        started_s=10.0,
        now_s=12.0,
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v377_pivot_blocked_no_arc_rft_eval_"
        "sudoku_flagged_skipped_games9_flagged_skipped4"
    )
    assert artifact["pivot_result"] == (
        "principle: ARC verifier-certified RFT training-time value was not measured; "
        "exp4079 landed only a blocked gate-check because exp4078 never launched training."
    )
    assert artifact["pivot_comparison"]["status"] == "blocked"
    assert artifact["pivot_comparison"]["rft_beats_cold_ci_excludes_zero"] is False
    assert artifact["pivot_comparison"]["rft_beats_or_matches_gold_sft"] is False
    assert artifact["sudoku_control_reproduced"] is False
    assert artifact["sudoku_control"]["status"] == "skipped_flagged"
    assert artifact["decentralization_distillation_outcome"].startswith(
        "unmeasured: no clean ARC RFT induction-rate delta"
    )
    assert artifact["games_solved_total"] == 9
    assert artifact["arc_accuracy"]["game_solved"] is True
    assert artifact["hardware_continuity"]["kv260_terminal_confirmed"] is True
    assert "gated_on" not in artifact

    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [
        4077,
        4078,
        4080,
        4083,
    ]
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == {4076, 4079, 4081, 4082, 4084}
    assert cited[4082] == {
        "experiment_id": 4082,
        "sha256": hashlib.sha256(
            Path("results/experiment_4082_ninth_game_explore_first.json").read_bytes()
        ).hexdigest(),
    }
    assert artifact["upstream_artifact_state"]["4079"]["included"] is True
    assert artifact["upstream_artifact_state"]["4080"]["skipped"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["games_solved_total"].startswith("BARE INT")


def test_req_capstone_4085_clean_fixture_records_decision_grade_rft(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4085: clean upstreams can satisfy the RFT and control headline."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(
        tmp_path, summary_statuses=_summary_statuses(), started_s=1.0, now_s=1.5
    )

    mod.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith(
        "success: capstone_v377_verifier_as_reward_rft_beats_cold_"
        "matches_gold_sudoku_reproduced_games10"
    )
    assert artifact["pivot_result"] == (
        "principle: verifier-certified RFT beat cold held-out with CI excluding zero "
        "and matched gold-SFT; the verifier's training-time value is measured."
    )
    assert artifact["pivot_comparison"]["rft_beats_cold_ci_excludes_zero"] is True
    assert artifact["pivot_comparison"]["rft_beats_or_matches_gold_sft"] is True
    assert artifact["sudoku_control_reproduced"] is True
    assert artifact["decentralization_distillation_outcome"] == (
        "moved_toward_codex: local induction 0.2600->0.4200 toward codex 0.5700"
    )
    assert artifact["games_solved_total"] == 10
    assert artifact["flagged_artifacts_skipped"] == []
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == set(
        mod.UPSTREAM_IDS
    )


def test_scenario_capstone_4085_write_artifact_validates_output(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4085: write_artifact emits the required deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4085_capstone_v377.json"),
        summary_statuses=_summary_statuses(),
        started_s=2.0,
        now_s=2.25,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4085
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4085_branch_states_are_explicit(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4085: missing, accumulating, and measured-null states stay explicit."""

    fallback = tmp_path / "results/experiment_4076_fallback.json"
    _write_json(fallback, {"honest_verdict": "complete: fallback"})
    paths = mod.selected_upstream_paths(tmp_path)
    statuses = mod.summarize_existing_artifacts(tmp_path, paths, {4076: {"returncode": 0}})
    assert paths[4076] == fallback
    assert paths[4077] is None
    assert statuses == {4076: {"returncode": 0}}

    assert mod.list_float_metric({"ci": "not-a-list"}, "ci") == []
    assert mod.nested_int(None, ("missing",)) == 0
    assert mod.nested_int({"outer": {"inner": 3}}, ("outer", "inner")) == 3
    assert mod.pivot_comparison(None, was_skipped=True)["status"] == "skipped_flagged"
    assert mod.pivot_comparison(None, was_skipped=False)["status"] == "missing"

    accumulating = mod.pivot_comparison(
        {"honest_verdict": "complete: train_accumulating", "training_accumulating": True},
        was_skipped=False,
    )
    assert accumulating["status"] == "accumulating"
    assert mod.pivot_result_text(accumulating).startswith(
        "principle: ARC verifier-certified RFT training is still accumulating"
    )
    assert mod.verdict(
        pivot=accumulating,
        sudoku={"status": "missing_or_blocked"},
        games_solved_total=9,
        skipped_count=0,
    ).startswith("complete: capstone_v377_pivot_train_accumulating")

    measured_null = mod.pivot_comparison(
        {
            "honest_verdict": "complete: measured_null",
            "rft_correct_induction_rate": 0.255,
            "cold_base_induction_rate": 0.25,
            "gold_sft_induction_rate": 0.26,
            "rft_vs_cold_delta": 0.005,
            "rft_vs_cold_ci95": [-0.02, 0.03],
            "rft_vs_gold_delta": -0.005,
            "rft_vs_gold_ci95": [-0.02, 0.01],
            "prior_local_induction_rate": 0.26,
            "codex_induction_rate": 0.57,
        },
        was_skipped=False,
    )
    assert measured_null["status"] == "measured_null"
    assert measured_null["rft_beats_or_matches_gold_sft"] is True
    assert mod.pivot_result_text(measured_null).startswith(
        "principle: verifier-certified RFT was measured on held-out ARC"
    )
    assert mod.decentralization_distillation_outcome(measured_null).startswith(
        "not_moved_toward_codex"
    )
    assert mod.verdict(
        pivot=measured_null,
        sudoku={"status": "failed"},
        games_solved_total=9,
        skipped_count=1,
    ).startswith("complete: capstone_v377_verifier_as_reward_rft_measured_null")
    assert mod.pivot_result_text({"status": "missing"}).startswith(
        "principle: ARC verifier-certified RFT training-time value was not measured"
    )

    clean = mod.build_artifact(
        tmp_path, summary_statuses={4076: {"returncode": 0}}, started_s=1.0, now_s=1.1
    )
    clean["inference_substrate"] = "wrong"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(clean)
