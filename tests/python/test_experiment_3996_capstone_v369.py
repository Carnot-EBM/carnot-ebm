"""Tests for Exp 3996 .369 GAP-4 confirm/decentralize/deploy capstone.

Spec refs: REQ-CAPSTONE-3996, SCENARIO-CAPSTONE-3996.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v369_3996 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_payloads() -> dict[int, JsonDict]:
    return {
        3988: {
            "honest_verdict": "success: precision_gate_confirmed",
            "primary_gate_passed": True,
            "agreement_is_selector_not_label": True,
            "n_gold_given_agreement": 64,
        },
        3987: {
            "honest_verdict": "complete: debiased_coverage_measured",
            "debiased_coverage_combined": 0.42,
        },
        3989: {
            "honest_verdict": "success: feedback_beats_redraw",
            "feedback_beats_redraw": True,
            "mcnemar_p": 0.031,
        },
        3991: {
            "honest_verdict": "complete: local_gguf_induced_below_codex",
            "real_local_gguf_inducer": True,
            "local_beats_vote": False,
            "local_induction_demo_perfect_rate": 1.0,
            "local_gated_pass2": 0.47,
            "cost_local_seconds": 19.5,
            "cost_codex_seconds_ref": 42.0,
        },
        3990: {
            "honest_verdict": "success: program_induction_stack_registered_reproduced",
            "verifier_registered": True,
            "arc2_reproduced_19of31": True,
            "arc1_reproduced_28of31": True,
            "gap5_entry_appended": True,
        },
        3992: {
            "honest_verdict": "success: verifier_validated_reinduction_advanced_r11l_to_L3",
            "new_levels_solved_this_task": 2,
            "ACCURACY_levels_solved": 3,
        },
        3993: {
            "honest_verdict": "complete: fourth_game_no_solve_pruner_rejected_unseen_dynamics",
            "game_solved": "none",
            "ACCURACY_levels_solved": 0,
            "real_env_confirmed": True,
        },
        3994: {
            "honest_verdict": "success: arcmemo_solve_transfer_v2_2668to17_actions",
            "solve_transfer_win": True,
            "actions_cold_start": 2668,
            "actions_with_memory": 17,
        },
        3995: {
            "honest_verdict": "complete: hardware_continuity_3995",
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
        experiment_id: {
            "returncode": 0,
            "stdout": f"summarized {experiment_id}",
            "stderr": "",
        }
        for experiment_id in ids
    }


def test_req_capstone_3996_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-3996: OpenSpec declares the .369 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-3996" in spec
    assert "SCENARIO-CAPSTONE-3996" in spec
    assert "gap4_decentralized" in spec
    assert "local_gated_pass2" in spec


def test_scenario_capstone_3996_current_missing_upstreams_remain_honest(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3996: missing exp3987-exp3991 are recorded, not gated."""

    payloads = {
        experiment_id: payload
        for experiment_id, payload in _artifact_payloads().items()
        if experiment_id >= 3992
    }
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=10.0,
        now_s=12.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_confirmed"] is False
    assert artifact["gap4_decentralized"] is False
    assert artifact["gap4_deployed"] is False
    assert artifact["local_generator_beats_vote"] is False
    assert artifact["total_games_solved"] == 3
    assert artifact["total_new_levels_this_milestone"] == 2
    assert artifact["arcmemo_solve_transfer_win"] is True
    assert artifact["arcmemo_actions_cold_start"] == 2668
    assert artifact["arcmemo_actions_with_memory"] == 17
    assert artifact["missing_upstream_artifacts"] == [
        {"experiment_id": 3987},
        {"experiment_id": 3988},
        {"experiment_id": 3989},
        {"experiment_id": 3990},
        {"experiment_id": 3991},
    ]
    assert artifact["flagged_artifacts_skipped"] == []
    assert artifact["duration_s"] == 2.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"]["gap4_confirmed"].startswith("BARE BOOL")

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == set(paths)
    assert cited[3992]["fields_imported"] == [
        "new_levels_solved_this_task",
        "ACCURACY_levels_solved",
    ]
    assert cited[3994]["sha256"] == hashlib.sha256(paths[3994].read_bytes()).hexdigest()
    assert artifact["upstream_artifact_state"]["3988"]["exists"] is False
    mod.validate_artifact(artifact)


def test_req_capstone_3996_all_axes_from_clean_upstreams(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3996: clean upstream metrics drive confirmed/decentralized/deployed axes."""

    payloads = _artifact_payloads()
    payloads[3993].update(
        {
            "honest_verdict": "success: fourth_game_solved",
            "game_solved": "su15-1944f8ab",
            "ACCURACY_levels_solved": 1,
            "real_env_confirmed": True,
        }
    )
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["gap4_confirmed"] is True
    assert artifact["confirmed_primary_gate_passed"] is True
    assert artifact["confirmed_agreement_is_selector_not_label"] is True
    assert artifact["confirmed_n_gold_given_agreement"] == 64
    assert artifact["confirmed_debiased_coverage_combined"] == 0.42
    assert artifact["confirmed_feedback_beats_redraw"] is True
    assert artifact["confirmed_mcnemar_p"] == 0.031
    assert artifact["gap4_decentralized"] is True
    assert artifact["local_generator_beats_vote"] is False
    assert artifact["local_induction_demo_perfect_rate"] == 1.0
    assert artifact["local_gated_pass2"] == 0.47
    assert artifact["local_cost_seconds"] == 19.5
    assert artifact["codex_cost_seconds_ref"] == 42.0
    assert artifact["gap4_deployed"] is True
    assert artifact["deployed_verifier_registered"] is True
    assert artifact["deployed_arc2_reproduced_19of31"] is True
    assert artifact["deployed_arc1_reproduced_28of31"] is True
    assert artifact["deployed_gap5_entry_appended"] is True
    assert artifact["total_games_solved"] == 4
    assert artifact["total_new_levels_this_milestone"] == 3
    assert artifact["fourth_game_solved"] is True


def test_req_capstone_3996_confidence_label_retirement_can_confirm(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3996: a clean confidence-label-only retirement is a confirmed answer."""

    payloads = _artifact_payloads()
    payloads[3988] = {
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


def test_req_capstone_3996_flagged_artifacts_are_excluded_and_count_false(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3996: flagged upstreams cannot satisfy milestone gates."""

    payloads = _artifact_payloads()
    payloads[3990] = {
        **payloads[3990],
        "flagged_adversarial": True,
        "verifier_registered": True,
        "arc2_reproduced_19of31": True,
        "arc1_reproduced_28of31": True,
    }
    paths = _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(paths)))

    assert artifact["gap4_deployed"] is False
    assert artifact["deployed_verifier_registered"] is False
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 3990,
            "path": "results/experiment_3990_fixture.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    assert 3990 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}


def test_req_capstone_3996_write_artifact_and_validate_rejects_regressions(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3996: artifact writing validates required bare fields."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(tuple(paths)),
        started_s=1.0,
        now_s=1.25,
    )

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema"] == "carnot.capstone_v369_3996.v1"
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
    bad["cited_upstream_artifacts"] = [{"experiment_id": 3992, "fields_imported": []}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)


def test_req_capstone_3996_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-3996: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / "results" / "experiment_3992_fixture.json"
    _write_json(path, _artifact_payloads()[3992])
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
    statuses = mod.summarize_existing_artifacts(tmp_path, {3992: path, 3993: None}, supplied=None)
    assert statuses == {
        3992: {
            "returncode": 0,
            "stdout": "summary for experiment_3992_fixture.json",
            "stderr": "",
        }
    }


def test_scenario_capstone_3996_script_wrapper_exists() -> None:
    """SCENARIO-CAPSTONE-3996: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_3996_capstone_v369.py")

    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "capstone_v369_3996" in text
    assert "write_artifact" in text
