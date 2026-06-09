"""Tests for Exp 3973 .367 ARC accuracy plus HONEST-efficiency capstone.

Spec refs: REQ-CAPSTONE-3973, SCENARIO-CAPSTONE-3973.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v367_3973 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_payloads() -> dict[int, JsonDict]:
    return {
        3964: {
            "honest_verdict": "complete: r11l_levels1_of6_first_fail2",
            "ACCURACY_levels_solved": 1,
            "new_levels_solved_this_task": 0,
            "level_summaries": [{"level": 1}, {"level": 2}],
            "real_env_confirmed": True,
            "duration_s": 0.77,
            "inference_substrate": "offline_arc_agi3_perception_planner_real_env_confirmed",
        },
        3965: {
            "honest_verdict": "complete: lp85_levels1_first_fail2",
            "ACCURACY_levels_solved": 1,
            "new_levels_solved_this_task": 0,
            "level_summaries": [{"level": 1}, {"level": 2}],
            "real_env_confirmed": True,
            "duration_s": 1.344,
            "inference_substrate": "offline_arc_agi3_perception_planner_real_env_confirmed",
        },
        3966: {
            "honest_verdict": "complete: third_game_solve_sc25-635fd71a_levels1_solvedTrue",
            "ACCURACY_levels_solved": 1,
            "game_solved": "sc25-635fd71a",
            "real_env_confirmed": True,
            "duration_s": 374.3,
            "inference_substrate": "offline_arc_agi3_perception_planner_real_env_confirmed",
        },
        3967: {
            "honest_verdict": "blocked_verifier_not_in_loop",
            "efficiency_ratio_with_over_without": 0.0,
            "verifier_invoked_in_loop": False,
            "actions_from_real_env": False,
            "cis_non_overlapping_pruner_helps": False,
            "duration_s": 0.0,
            "inference_substrate": "offline_air_gapped_arc_agi3_local_environments",
        },
        3968: {
            "honest_verdict": "complete: exp3968_active_codex_nonspatial_sweep_trustworthy_0of6",
            "n_trustworthy_at_0.15": 0,
            "n_games": 6,
            "duration_s": 1082.4,
            "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
        },
        3969: {
            "honest_verdict": "complete: pinductor_latents_no_drop_energy",
            "energy_drop": 0.003625,
            "positive_control_passed": True,
            "duration_s": 28.94,
            "inference_substrate": "offline_arc_agi3_pinductor",
        },
        3970: {
            "honest_verdict": "success: arcmemo_transfer_win_reused_2_later_games",
            "transfer_win": True,
            "duration_s": 0.4,
            "inference_substrate": "offline_arc_agi3_existing_codex_sweep_plus_arcmemo_concept_memory",
        },
        3971: {
            "honest_verdict": "success: quota_gate_cleared_hybrid_levels3_baseline0_prior0_operator_ready",
            "quota_gate_cleared": True,
            "duration_s": 16.843,
            "inference_substrate": "offline_arc_agi3_hybrid_policy_quota_gate_local_env",
        },
    }


def _write_artifacts(root: Path, payloads: dict[int, JsonDict]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for experiment_id, payload in payloads.items():
        path = root / "results" / f"experiment_{experiment_id}_fixture.json"
        _write_json(path, payload)
        paths[experiment_id] = path
    return paths


def _summary_statuses(ids: list[int] | tuple[int, ...] = mod.UPSTREAM_IDS) -> dict[int, JsonDict]:
    return {
        experiment_id: {
            "returncode": 0,
            "stdout": f"summarized {experiment_id}",
            "stderr": "",
        }
        for experiment_id in ids
    }


def test_req_capstone_3973_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-3973: OpenSpec declares the .367 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")
    assert "REQ-CAPSTONE-3973" in spec
    assert "SCENARIO-CAPSTONE-3973" in spec
    assert "flagged_adversarial:true" in spec


def test_scenario_capstone_3973_clean_upstreams_aggregate_headlines(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-3973: existing clean upstreams produce the honest headline."""

    paths = _write_artifacts(tmp_path, _artifact_payloads())
    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(),
        started_s=10.0,
        now_s=12.5,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["total_real_levels_solved"] == 3
    assert artifact["accuracy_progress_vs_v366_baseline"] == 1
    assert artifact["third_game_solved"] is True
    assert artifact["verifier_earns_efficiency_on_real_games"] is False
    assert artifact["m3_artifact_clean_not_flagged"] is False
    assert artifact["n_trustworthy_world_models"] == 0
    assert artifact["hidden_state_fixed"] is True
    assert artifact["transfer_win"] is True
    assert artifact["quota_gate_cleared"] is True
    assert artifact["flagged_artifacts_skipped"] == []
    assert artifact["missing_upstream_artifacts"] == []
    assert artifact["duration_s"] == 2.5
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == set(mod.UPSTREAM_IDS)
    assert cited[3967]["fields_imported"] == [
        "efficiency_ratio_with_over_without",
        "verifier_invoked_in_loop",
        "actions_from_real_env",
        "cis_non_overlapping_pruner_helps",
    ]
    assert cited[3964]["sha256"] == hashlib.sha256(paths[3964].read_bytes()).hexdigest()
    assert artifact["field_principles"]["total_real_levels_solved"].startswith("BARE INT")
    mod.validate_artifact(artifact)


def test_req_capstone_3973_flagged_artifact_is_skipped_before_import(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3973: flagged M3 metrics cannot satisfy the efficiency verdict."""

    payloads = _artifact_payloads()
    payloads[3967] = {
        **payloads[3967],
        "flagged_adversarial": True,
        "efficiency_ratio_with_over_without": 7.0,
        "verifier_invoked_in_loop": True,
        "actions_from_real_env": True,
        "cis_non_overlapping_pruner_helps": True,
        "honest_verdict": "success: fabricated_clean_m3",
    }
    _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses())

    assert artifact["verifier_earns_efficiency_on_real_games"] is False
    assert artifact["m3_artifact_clean_not_flagged"] is False
    assert artifact["flagged_artifacts_skipped"] == [
        {
            "experiment_id": 3967,
            "path": "results/experiment_3967_fixture.json",
            "reason": "flagged_adversarial:true",
        }
    ]
    assert 3967 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}


def test_req_capstone_3973_missing_upstream_is_recorded_not_blocking(tmp_path: Path) -> None:
    """REQ-CAPSTONE-3973: missing upstream artifacts are missing state, not a gate."""

    payloads = _artifact_payloads()
    del payloads[3968]
    _write_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(tmp_path, summary_statuses=_summary_statuses(tuple(payloads)))

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 3968}]
    assert artifact["n_trustworthy_world_models"] == 0
    assert artifact["upstream_artifact_state"]["3968"]["exists"] is False


def test_req_capstone_3973_write_artifact_and_validate_rejects_regressions(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-3973: artifact writing validates required bare fields."""

    _write_artifacts(tmp_path, _artifact_payloads())
    output = mod.write_artifact(
        tmp_path,
        summary_statuses=_summary_statuses(),
        started_s=1.0,
        now_s=1.125,
    )

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema"] == "carnot.capstone_v367_3973.v1"
    assert mod.is_sha256(written["reproducibility_checksum"])

    bad = dict(written)
    bad["honest_verdict"] = "maybe"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["total_real_levels_solved"] = True
    with pytest.raises(ValueError, match="bare int"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["third_game_solved"] = "yes"
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(bad)

    bad = dict(written)
    bad["cited_upstream_artifacts"] = [{"experiment_id": 3964, "fields_imported": []}]
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(bad)


def test_req_capstone_3973_summary_runner_uses_mandated_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-3973: upstream reading shells through summarize_artifact.py."""

    path = tmp_path / "results" / "experiment_3964_fixture.json"
    _write_json(path, _artifact_payloads()[3964])
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
    statuses = mod.summarize_existing_artifacts(tmp_path, {3964: path, 3965: None}, supplied=None)
    assert statuses == {
        3964: {
            "returncode": 0,
            "stdout": "summary for experiment_3964_fixture.json",
            "stderr": "",
        }
    }


def test_scenario_capstone_3973_script_wrapper_exists() -> None:
    """SCENARIO-CAPSTONE-3973: the requested experiment entrypoint exists."""

    script = Path("scripts/experiments/experiment_3973_capstone_v367.py")
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "capstone_v367_3973" in text
    assert "write_artifact" in text
