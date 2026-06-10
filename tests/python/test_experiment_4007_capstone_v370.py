"""Tests for Exp 4007 .370 GAP-4 confirm/decentralize/deploy capstone.

Spec refs: REQ-CAPSTONE-4007, SCENARIO-CAPSTONE-4007.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v370_4007 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_payloads() -> dict[int, JsonDict]:
    return {
        3997: {
            "honest_verdict": "success: poison_guard_green",
            "pretest_suite_green": True,
            "quarantined_tests": ["test_poison_guard.py::test_old_cascade"],
        },
        3998: {
            "honest_verdict": "complete: gap4_deselection_coverage_0.4091_n11",
            "debiased_coverage_combined": 0.6304,
        },
        3999: {
            "honest_verdict": "complete: protocol_preregistered_pending_execution",
            "primary_gate_passed": False,
            "agreement_is_selector_not_label": False,
            "n_gold_given_agreement": 0,
        },
        4000: {
            "honest_verdict": "complete: feedback_no_better_than_redraw_p1.0_FALSE_NEGATIVE_RISK",
            "feedback_beats_redraw": False,
            "mcnemar_p": 1.0,
        },
        4001: {
            "honest_verdict": "success: gap4_stack_registered_arc2_19of31_arc1_28of31_reproduced",
            "verifier_registered": True,
            "arc2_reproduced_19of31": True,
            "arc1_reproduced_28of31": True,
            "gap5_entry_appended": True,
        },
        4002: {
            "honest_verdict": "complete: gap4_local_induction0.2581_pass20.4516_below_codex",
            "inference_substrate": "live_llm_inference",
            "local_beats_vote": False,
            "local_induction_demo_perfect_rate": 0.2581,
            "local_gated_pass2": 0.4516,
            "cost_local_seconds": 60.16,
            "cost_codex_seconds_ref": 46.24,
            "local_model_used": "gemma-4-26B-A4B",
            "model_specs": {"generator_gguf_path": "/tmp/model.gguf"},
            "preconditions_checked": [
                {"resource": "local_gguf_cached", "available": True},
                {"resource": "llama_cpp", "available": True},
                {"resource": "eval_pool", "available": True},
            ],
            "verifier_side_unchanged": True,
        },
        4003: {
            "honest_verdict": "complete: level_frontier_holds_total5",
            "new_levels_this_task": 0,
            "ACCURACY_total_levels_solved": 5,
        },
        4004: {
            "honest_verdict": "success: fourth_game_solved_su15-1944f8ab_at_action14",
            "game_solved": "su15-1944f8ab",
            "ACCURACY_levels_solved": 1,
        },
        4005: {
            "honest_verdict": "success: arcmemo_solve_transfer_v3_14to10_actions",
            "solve_transfer_win": True,
            "actions_cold_start": 14,
            "actions_with_memory": 10,
        },
        4006: {
            "honest_verdict": "complete: hardware_continuity_4006",
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


def test_req_capstone_4007_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4007: OpenSpec declares the .370 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4007" in spec
    assert "SCENARIO-CAPSTONE-4007" in spec
    assert "gap4_phase_ran_this_time" in spec
    assert "total_levels_solved" in spec


def test_scenario_capstone_4007_current_missing_poison_guard_remains_honest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4007: missing exp3997 is recorded while clean phase artifacts aggregate."""

    payloads = {experiment_id: payload for experiment_id, payload in _artifact_payloads().items() if experiment_id != 3997}
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=10.0,
        now_s=12.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["gap4_phase_ran_this_time"] is True
    assert artifact["gap4_confirmed"] is False
    assert artifact["gap4_decentralized"] is True
    assert artifact["gap4_deployed"] is True
    assert artifact["local_generator_beats_vote"] is False
    assert artifact["total_games_solved"] == 4
    assert artifact["total_levels_solved"] == 5
    assert artifact["arcmemo_solve_transfer_win"] is True
    assert artifact["arcmemo_actions_cold_start"] == 14
    assert artifact["arcmemo_actions_with_memory"] == 10
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 3997}]
    assert artifact["flagged_artifacts_skipped"] == []
    assert artifact["duration_s"] == 2.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["gap4_phase_ran_this_time"].startswith("BARE BOOL")

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == set(paths)
    assert cited[4003]["fields_imported"] == ["new_levels_this_task", "ACCURACY_total_levels_solved"]
    assert cited[4005]["sha256"] == hashlib.sha256(paths[4005].read_bytes()).hexdigest()
    assert artifact["upstream_artifact_state"]["3997"]["exists"] is False
    mod.validate_artifact(artifact)


def test_req_capstone_4007_all_axes_from_clean_upstreams(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4007: clean upstream metrics drive confirmed/decentralized/deployed axes."""

    payloads = _artifact_payloads()
    payloads[3999].update(
        {
            "honest_verdict": "success: precision_gate_confirmed",
            "primary_gate_passed": True,
            "agreement_is_selector_not_label": True,
            "n_gold_given_agreement": 64,
        }
    )
    payloads[4002]["local_beats_vote"] = True
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["gap4_phase_ran_this_time"] is True
    assert artifact["poison_guard_pretest_suite_green"] is True
    assert artifact["poison_guard_quarantined_tests"] == ["test_poison_guard.py::test_old_cascade"]
    assert artifact["gap4_confirmed"] is True
    assert artifact["confirmed_primary_gate_passed"] is True
    assert artifact["confirmed_agreement_is_selector_not_label"] is True
    assert artifact["confirmed_n_gold_given_agreement"] == 64
    assert artifact["confirmed_debiased_coverage_combined"] == 0.6304
    assert artifact["confirmed_feedback_beats_redraw"] is False
    assert artifact["confirmed_mcnemar_p"] == 1.0
    assert artifact["gap4_decentralized"] is True
    assert artifact["local_generator_beats_vote"] is True
    assert artifact["local_induction_demo_perfect_rate"] == 0.2581
    assert artifact["local_gated_pass2"] == 0.4516
    assert artifact["local_cost_seconds"] == 60.16
    assert artifact["codex_cost_seconds_ref"] == 46.24
    assert artifact["local_model_used"] == "gemma-4-26B-A4B"
    assert artifact["gap4_deployed"] is True
    assert artifact["deployed_verifier_registered"] is True
    assert artifact["deployed_arc2_reproduced_19of31"] is True
    assert artifact["deployed_arc1_reproduced_28of31"] is True
    assert artifact["deployed_gap5_entry_appended"] is True
    assert artifact["total_games_solved"] == 4
    assert artifact["total_levels_solved"] == 5
    assert artifact["new_levels_this_task"] == 0
    assert artifact["fourth_game_accuracy_levels_solved"] == 1


def test_req_capstone_4007_confidence_label_retirement_can_confirm(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4007: a clean confidence-label-only retirement is a confirmed answer."""

    payloads = _artifact_payloads()
    payloads[3999] = {
        "honest_verdict": "complete: confidence_label_only_retired_cleanly",
        "primary_gate_passed": False,
        "confidence_label_only_retired": True,
        "agreement_is_selector_not_label": False,
        "n_gold_given_agreement": 0,
    }
    _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(payloads)))

    assert artifact["gap4_confirmed"] is True
    assert artifact["confirmed_primary_gate_passed"] is False
    assert artifact["confirmed_confidence_label_only_retired"] is True


def test_req_capstone_4007_flagged_artifacts_are_excluded_and_count_false(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4007: flagged upstreams cannot satisfy milestone gates."""

    payloads = _artifact_payloads()
    payloads[4002] = {
        **payloads[4002],
        "flagged_adversarial": True,
        "local_beats_vote": True,
    }
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["gap4_decentralized"] is False
    assert artifact["local_generator_beats_vote"] is False
    assert artifact["local_model_used"] == ""
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 4002,
            "path": "results/experiment_4002_fixture.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    assert 4002 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert mod.precondition_available({"preconditions_checked": "not-a-list"}, "llama_cpp") is False
    assert mod.precondition_available({"preconditions_checked": [{"resource": "other", "available": True}]}, "llama_cpp") is False


def test_req_capstone_4007_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4007: artifact writing validates required bare fields."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=1.0,
        now_s=1.25,
    )

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema"] == "carnot.capstone_v370_4007.v1"
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["total_levels_solved"] = True
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["gap4_phase_ran_this_time"] = "false"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 4003, "fields_imported": []}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)


def test_req_capstone_4007_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4007: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / "results" / "experiment_4003_fixture.json"
    _write_json(path, _artifact_payloads()[4003])
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["text"] is True
        assert kwargs["capture_output"] is True
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
    statuses = mod.summarize_existing_artifacts(tmp_path, {4003: path, 4004: None}, supplied=None)
    assert statuses == {
        4003: {
            "returncode": 0,
            "stdout": "summary for experiment_4003_fixture.json",
            "stderr": "",
        }
    }


def test_scenario_capstone_4007_script_wrapper_exists() -> None:
    """SCENARIO-CAPSTONE-4007: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_4007_capstone_v370.py")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "capstone_v370_4007" in text
    assert "write_artifact" in text
