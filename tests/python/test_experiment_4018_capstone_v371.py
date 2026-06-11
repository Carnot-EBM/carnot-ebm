"""Tests for Exp 4018 .371 GAP-4 precision/decentralization capstone.

Spec refs: REQ-CAPSTONE-4018, SCENARIO-CAPSTONE-4018.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v371_4018 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_payloads() -> dict[int, JsonDict]:
    return {
        4008: {
            "honest_verdict": "success: poison_guard_green",
            "pretest_suite_green": True,
            "quarantined_tests": ["tests/python/test_old_cascade.py::test_poisoned"],
        },
        4009: {
            "honest_verdict": "success: precision_selector_confirmed",
            "execution_floor_met": True,
            "primary_gate_passed": True,
            "agreement_is_selector_not_label": True,
            "n_gold_given_agreement": 64,
            "total_codex_calls": 192,
            "n_agreement_events": 80,
        },
        4010: {
            "honest_verdict": "success: cross_example_selector_helped",
            "selector_beats_output_agreement": True,
        },
        4011: {
            "honest_verdict": "success: feedback_beats_redraw",
            "feedback_beats_redraw": True,
            "n_discordant_pairs": 7,
        },
        4012: {
            "honest_verdict": "success: local_bestofn_beat_vote",
            "local_beats_vote": True,
            "local_demo_perfect_coverage_bestofn": 0.61,
            "coverage_gain_vs_3attempt": 0.08,
            "local_gated_pass2": 0.58,
            "cost_local_seconds": 38.0,
            "cost_codex_seconds_ref": 46.24,
        },
        4013: {
            "honest_verdict": "success: verifier_cheaper_than_judge",
            "selection_accuracy_parity": True,
            "cost_ratio_judge_over_verifier": 95.25,
        },
        4014: {
            "honest_verdict": "success: wall_broken_total6",
            "new_levels_this_task": 1,
            "ACCURACY_total_levels_solved": 6,
        },
        4015: {
            "honest_verdict": "success: fifth_game_solved",
            "game_solved": "tn36-ef4dde99",
            "ACCURACY_levels_solved": 1,
        },
        4016: {
            "honest_verdict": "success: arcmemo_transfer",
            "solve_transfer_win": True,
            "actions_cold_start": 11,
            "actions_with_memory": 7,
        },
        4017: {
            "honest_verdict": "complete: hardware_continuity_recorded",
        },
    }


def _write_artifacts(root: Path, payloads: dict[int, JsonDict]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for experiment_id, payload in payloads.items():
        path = root / "results" / f"experiment_{experiment_id}_fixture.json"
        _write_json(path, payload)
        paths[experiment_id] = path
    return paths


def _summary_statuses(ids: list[int] | tuple[int, ...]) -> dict[int, JsonDict]:
    return {
        experiment_id: {"returncode": 0, "stdout": f"summarized {experiment_id}", "stderr": ""}
        for experiment_id in ids
    }


def test_req_capstone_4018_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4018: OpenSpec declares the .371 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4018" in spec
    assert "SCENARIO-CAPSTONE-4018" in spec
    assert "confirmation_executed_this_time" in spec
    assert "gap4_decentralization_effective" in spec


def test_scenario_capstone_4018_current_artifacts_honestly_aggregate() -> None:
    """SCENARIO-CAPSTONE-4018: current landed artifacts produce the honest .371 verdict."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses={
            4008: {"returncode": 0, "stdout": "summary 4008", "stderr": ""},
            4009: {"returncode": 0, "stdout": "summary 4009", "stderr": ""},
            4010: {"returncode": 0, "stdout": "summary 4010", "stderr": ""},
            4012: {"returncode": 1, "stdout": "summary 4012", "stderr": ""},
            4013: {"returncode": 2, "stdout": "summary 4013", "stderr": ""},
            4014: {"returncode": 0, "stdout": "summary 4014", "stderr": ""},
            4015: {"returncode": 0, "stdout": "summary 4015", "stderr": ""},
            4016: {"returncode": 0, "stdout": "summary 4016", "stderr": ""},
            4017: {"returncode": 0, "stdout": "summary 4017", "stderr": ""},
        },
        started_s=3.0,
        now_s=5.5,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["confirmation_executed_this_time"] is False
    assert artifact["gap4_confirmed"] is False
    assert artifact["gap4_decentralization_effective"] is False
    assert artifact["local_generator_beats_vote"] is False
    assert artifact["cross_example_selector_helped"] is False
    assert artifact["verifier_cheaper_than_judge"] is False
    assert artifact["total_games_solved"] == 5
    assert artifact["total_levels_solved"] == 5
    assert artifact["arcmemo_solve_transfer_win"] is True
    assert artifact["pretest_suite_green"] is False
    assert artifact["confirmation_total_codex_calls"] == 0
    assert artifact["confirmation_n_agreement_events"] == 0
    assert artifact["local_coverage_gain_vs_3attempt"] == 0.0
    assert artifact["efficiency_cost_ratio_judge_over_verifier"] == 0.0
    assert artifact["fifth_game_solved"] is True
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 4011}]
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 4013,
            "path": "results/experiment_4013_verifier_vs_judge_efficiency.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    assert artifact["duration_s"] == 2.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["gap4_confirmed"].startswith("BARE BOOL")

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert 4013 not in cited
    assert set(cited) == {4008, 4009, 4010, 4012, 4014, 4015, 4016, 4017}
    assert cited[4009]["fields_imported"] == [
        "execution_floor_met",
        "primary_gate_passed",
        "confidence_label_only_retired",
        "agreement_is_selector_not_label",
        "n_gold_given_agreement",
        "total_codex_calls",
        "n_agreement_events",
    ]
    assert cited[4015]["sha256"] == hashlib.sha256(
        Path("results/experiment_4015_fifth_game_explore_first.json").read_bytes(),
    ).hexdigest()
    assert artifact["upstream_artifact_state"]["4011"]["exists"] is False
    assert artifact["upstream_artifact_state"]["4013"]["included"] is False
    mod.validate_artifact(artifact)


def test_req_capstone_4018_all_axes_from_clean_upstreams(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4018: clean upstream metrics drive all headline booleans."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["confirmation_executed_this_time"] is True
    assert artifact["gap4_confirmed"] is True
    assert artifact["confirmed_execution_floor_met"] is True
    assert artifact["confirmed_primary_gate_passed"] is True
    assert artifact["confirmed_agreement_is_selector_not_label"] is True
    assert artifact["confirmed_n_gold_given_agreement"] == 64
    assert artifact["confirmation_total_codex_calls"] == 192
    assert artifact["confirmation_n_agreement_events"] == 80
    assert artifact["cross_example_selector_helped"] is True
    assert artifact["feedback_beats_redraw"] is True
    assert artifact["feedback_n_discordant_pairs"] == 7
    assert artifact["gap4_decentralization_effective"] is True
    assert artifact["local_generator_beats_vote"] is True
    assert artifact["local_demo_perfect_coverage_bestofn"] == 0.61
    assert artifact["local_coverage_gain_vs_3attempt"] == 0.08
    assert artifact["local_gated_pass2"] == 0.58
    assert artifact["local_cost_seconds"] == 38.0
    assert artifact["codex_cost_seconds_ref"] == 46.24
    assert artifact["verifier_cheaper_than_judge"] is True
    assert artifact["efficiency_selection_accuracy_parity"] is True
    assert artifact["efficiency_cost_ratio_judge_over_verifier"] == 95.25
    assert artifact["total_games_solved"] == 5
    assert artifact["total_levels_solved"] == 6
    assert artifact["new_levels_this_task"] == 1
    assert artifact["fifth_game_solved"] is True
    assert artifact["fifth_game_accuracy_levels_solved"] == 1
    assert artifact["arcmemo_solve_transfer_win"] is True
    assert artifact["arcmemo_actions_cold_start"] == 11
    assert artifact["arcmemo_actions_with_memory"] == 7
    assert artifact["pretest_suite_green"] is True
    assert artifact["poison_guard_quarantined_tests"] == ["tests/python/test_old_cascade.py::test_poisoned"]


def test_req_capstone_4018_confidence_label_retirement_can_confirm(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4018: a clean powered confidence-label-only retirement can confirm."""

    payloads = _artifact_payloads()
    payloads[4009] = {
        "honest_verdict": "complete: confidence_label_only_retired_cleanly",
        "execution_floor_met": True,
        "primary_gate_passed": False,
        "confidence_label_only_retired": True,
        "agreement_is_selector_not_label": False,
        "n_gold_given_agreement": 0,
        "total_codex_calls": 32,
        "n_agreement_events": 9,
    }
    _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(payloads)))

    assert artifact["confirmation_executed_this_time"] is True
    assert artifact["gap4_confirmed"] is True
    assert artifact["confirmed_primary_gate_passed"] is False
    assert artifact["confirmed_confidence_label_only_retired"] is True


def test_req_capstone_4018_decentralization_gap_can_close_without_strong_vote_win(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4018: coverage gain plus high local pass2 is the honest closing-gap form."""

    payloads = _artifact_payloads()
    payloads[4012]["local_beats_vote"] = False
    payloads[4012]["coverage_gain_vs_3attempt"] = 0.04
    payloads[4012]["local_gated_pass2"] = mod.LOCAL_PASS2_APPROACHING_CODEX_FLOOR
    _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(payloads)))

    assert artifact["local_generator_beats_vote"] is False
    assert artifact["gap4_decentralization_effective"] is True


def test_req_capstone_4018_flagged_or_blocked_artifacts_count_false(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4018: flagged and blocked upstreams cannot satisfy owed gates."""

    payloads = _artifact_payloads()
    payloads[4009] = {
        **payloads[4009],
        "honest_verdict": "blocked_execution_floor_unmet",
        "execution_floor_met": True,
    }
    payloads[4012] = {
        **payloads[4012],
        "flagged_adversarial": True,
        "local_beats_vote": True,
    }
    payloads[4013] = {
        **payloads[4013],
        "flagged_adversarial": True,
    }
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["confirmation_executed_this_time"] is False
    assert artifact["gap4_confirmed"] is False
    assert artifact["gap4_decentralization_effective"] is False
    assert artifact["local_generator_beats_vote"] is False
    assert artifact["verifier_cheaper_than_judge"] is False
    assert artifact["local_cost_seconds"] == 0.0
    assert [row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]] == [4012, 4013]
    assert 4012 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}


def test_req_capstone_4018_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4018: artifact writing validates the required bare fields."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=1.0,
        now_s=1.25,
    )

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema"] == "carnot.capstone_v371_4018.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["total_games_solved"] = True
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["gap4_confirmed"] = "false"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["duration_s"] = False
    with pytest.raises(ValueError, match="bare number"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["inference_substrate"] = 371
    with pytest.raises(ValueError, match="string"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4015, "fields_imported": []}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["flagged_artifacts_skipped"] = {}
    with pytest.raises(ValueError, match="list"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["reproducibility_checksum"] = "not-sha"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)


def test_req_capstone_4018_helpers_and_summary_runner_use_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4018: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / "results" / "experiment_4014_fixture.json"
    _write_json(path, _artifact_payloads()[4014])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(command, 0, stdout="summary", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    status = mod.run_summarize_artifact(tmp_path, path)

    assert status == {"returncode": 0, "stdout": "summary", "stderr": ""}
    assert calls == [[str(mod.PYTHON_BIN), "scripts/summarize_artifact.py", str(path)]]

    monkeypatch.setattr(
        mod,
        "run_summarize_artifact",
        lambda root, artifact_path: {
            "returncode": 0,
            "stdout": f"summary for {artifact_path.name}",
            "stderr": "",
        },
    )
    statuses = mod.summarize_existing_artifacts(tmp_path, {4014: path, 4015: None}, supplied=None)
    assert statuses == {
        4014: {
            "returncode": 0,
            "stdout": "summary for experiment_4014_fixture.json",
            "stderr": "",
        }
    }
    assert mod.summarize_existing_artifacts(tmp_path, {4014: path}, supplied={4014: {"returncode": 2}}) == {
        4014: {"returncode": 2}
    }
    assert mod.float_metric({"x": True}, "x") == 0.0
    assert mod.float_metric({"x": 2}, "x") == 2.0
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.list_metric({"x": "not-list"}, "x") == []
    assert mod.is_sha256("f" * 64) is True
    assert mod.is_sha256("z" * 64) is False


def test_scenario_capstone_4018_script_wrapper_exists() -> None:
    """SCENARIO-CAPSTONE-4018: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_4018_capstone_v371.py")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "capstone_v371_4018" in text
    assert "write_artifact" in text
