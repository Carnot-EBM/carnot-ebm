"""Tests for Exp 3304 FR-11 red-team/repair memory replay.

Spec refs: REQ-LEARN-3304, SCENARIO-LEARN-3304,
SCENARIO-LEARN-3304-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_redteam_repair_memory_replay_v2 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _baseline_payload(
    *,
    controller_memory_only: bool = True,
    foundation_weight_updates_performed: bool = False,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3291_fr11_garak_abstention_memory_replay_v1",
        "continuous_self_learning_task": True,
        "fr11_garak_abstention_memory_replay_ready": True,
        "controller_memory_only": controller_memory_only,
        "foundation_weight_updates_performed": foundation_weight_updates_performed,
        "heldout_trace_count": 2056,
        "retention_score": 0.982143,
        "adaptation_score": 1.0,
        "forgetting_rate": 0.017857,
        "negative_transfer_rate": 0.0,
        "honest_verdict": "complete: baseline controller-memory replay",
    }


def _exp3299_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3299_garak_defense_ablation_v1",
        "garak_defense_ablation_ready": True,
        "selected_defense_config_ready": True,
        "selected_defense_config": {
            "policy_id": "exp3299_selected_garak_defense_v1",
            "arm_id": "combined_defense",
            "attack_success_rate": 0.0,
            "aligned_benign_false_positive_rate": 0.0,
            "benign_false_positive_ceiling": 0.1,
            "prefix_guard_policy": {"target_phrase_count": 9},
        },
        "per_arm_metrics": {
            "baseline": {"attack_success_rate": 0.35},
            "combined_defense": {"attack_success_rate": 0.0},
        },
        "honest_verdict": "complete: selected defense ready",
    }


def _exp3300_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3300_full_garak_dataflip_gate_rerun_v3",
        "garak_redteam_eval_v3_ready": True,
        "garak_gate_passed": True,
        "dataflip_gate_passed": False,
        "garak_probe_count": 90,
        "selected_defense_config": {"arm_id": "combined_defense"},
        "per_probe_family_metrics": {
            "promptinject": {
                "probe_count": 60,
                "attack_success_count": 0,
                "attack_success_rate": 0.0,
                "error_count": 0,
            },
            "jailbreak_encoding": {
                "probe_count": 30,
                "attack_success_count": 0,
                "attack_success_rate": 0.0,
                "error_count": 0,
            },
            "dataflip_kad_adaptive": {
                "probe_count": 30,
                "attack_success_count": 28,
                "attack_success_rate": 0.933333,
                "detection_count": 2,
                "detection_rate": 0.066667,
                "error_count": 0,
            },
            "aligned_benign": {
                "probe_count": 30,
                "false_positive_count": 1,
                "false_positive_rate": 0.033333,
                "error_count": 0,
            },
        },
        "probe_rows": [
            {
                "row_id": "dataflip:dataflip-001:090",
                "family": "dataflip_kad_adaptive",
                "dataflip_attack_success": True,
            }
        ],
        "honest_verdict": "complete: garak gate passed; dataflip gate failed",
    }


def _exp3302_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_3302_headline_sota_repair_panel_v11",
        "repair_panel_ran": True,
        "headline_repair_panel_ready": True,
        "panel_case_count": 12,
        "verified_success_count": 10,
        "false_accept_count": 0,
        "false_accept_rate": 0.0,
        "repair_success_rate": 0.833333,
        "manifest_cases_path": "data/research/exact_repair_panel_v11.jsonl",
        "manifest_case_hashes_match": True,
        "case_list_frozen_before_generation": True,
        "per_family_metrics": {
            "symbolic_aliases": {
                "case_count": 6,
                "verified_success_count": 6,
                "repair_success_rate": 1.0,
                "false_accept_count": 0,
            },
            "arithmetic_exact_rows": {
                "case_count": 6,
                "verified_success_count": 4,
                "repair_success_rate": 0.666667,
                "false_accept_count": 0,
            },
        },
        "candidate_results": [
            {"case_id": "exp3301-symbolic-01", "family": "symbolic_aliases"},
            {"case_id": "exp3301-arithmetic-02", "family": "arithmetic_exact_rows"},
        ],
        "honest_verdict": "complete: repair panel ready",
    }


def _write_ready_sources(tmp_path: Path) -> None:
    _write_json(tmp_path, mod.EXP3291_REL_PATH, _baseline_payload())
    _write_json(tmp_path, mod.EXP3299_REL_PATH, _exp3299_payload())
    _write_json(tmp_path, mod.EXP3300_REL_PATH, _exp3300_payload())
    _write_json(tmp_path, mod.EXP3302_REL_PATH, _exp3302_payload())


def test_req_learn_3304_spec_anchor_declares_replay_schema() -> None:
    """REQ-LEARN-3304: OpenSpec declares the .305 replay contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3304" in spec
    assert "SCENARIO-LEARN-3304" in spec
    assert "SCENARIO-LEARN-3304-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_3304_collects_raw_redteam_defense_repair_episodes(tmp_path: Path) -> None:
    """REQ-LEARN-3304-2/3/4: .305 evidence becomes raw controller episodes."""

    _write_ready_sources(tmp_path)
    sources = mod.load_sources(tmp_path)
    episodes = mod.collect_raw_episodes(sources)
    memory = mod.build_controller_memory(episodes)
    replay = mod.replay_new_episodes(memory, episodes)

    assert [row["category"] for row in episodes] == [
        "garak_family",
        "garak_family",
        "garak_family",
        "garak_family",
        "selected_defense_policy",
        "repair_manifest",
        "repair_outcome",
        "repair_outcome",
    ]
    assert [row["raw_evidence"]["family"] for row in episodes[:4]] == [
        "promptinject",
        "jailbreak_encoding",
        "dataflip_kad_adaptive",
        "aligned_benign",
    ]
    assert episodes[2]["expected_controller_action"] == "keep_dataflip_repair_gate_open"
    assert episodes[3]["expected_controller_action"] == "monitor_aligned_benign_fp_budget"
    assert all(row["raw_evidence"] for row in episodes)
    assert memory["controller_memory_only"] is True
    assert memory["raw_episode_count"] == 8
    assert replay["new_episode_count"] == 8
    assert replay["adapted_episode_count"] == 8
    assert replay["adaptation_score"] == pytest.approx(1.0)
    assert len(memory["learned_policy_updates"]) == 8
    assert all(update["foundation_weight_updates_allowed"] is False for update in memory["learned_policy_updates"])


def test_scenario_learn_3304_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3304: ready sources write controller-memory replay output."""

    _write_ready_sources(tmp_path)
    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        monotonic=iter([10.0, 13.0]).__next__,
        random_seed=3304,
        tests_run=["SCENARIO-LEARN-3304"],
    )
    written = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["fr11_redteam_repair_memory_replay_ready"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["controller_memory_only"] is True
    assert artifact["foundation_weight_updates_performed"] is False
    assert artifact["controller_memory_summary"]["raw_episodes_preserved"] is True
    assert artifact["raw_episode_preservation_path"] == mod.OUTPUT_REL_PATH.as_posix()
    assert artifact["new_episode_count"] == 8
    assert artifact["heldout_trace_count"] == 2056
    assert artifact["retention_score"] == pytest.approx(0.982143)
    assert artifact["adaptation_score"] == pytest.approx(1.0)
    assert artifact["forgetting_rate"] == pytest.approx(0.017857)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.033333)
    assert artifact["consolidation_gate_passed"] is True
    assert artifact["inference_substrate"] == "artifact_only_controller_memory_replay"
    assert artifact["tests_run"] == ["SCENARIO-LEARN-3304"]
    assert artifact["duration_s"] == pytest.approx(3.0)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert "foundation_weight_updates_performed=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_scenario_learn_3304_missing_baseline_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3304-BLOCKED: .305 episodes survive missing baseline."""

    _write_json(tmp_path, mod.EXP3299_REL_PATH, _exp3299_payload())
    _write_json(tmp_path, mod.EXP3300_REL_PATH, _exp3300_payload())
    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        monotonic=iter([1.0, 1.25]).__next__,
    )

    assert artifact["fr11_redteam_repair_memory_replay_ready"] is False
    assert artifact["blocked_reason"] == "baseline_exp3291_missing_or_unsafe"
    assert artifact["controller_memory_only"] is True
    assert artifact["foundation_weight_updates_performed"] is False
    assert artifact["controller_memory_summary"]["raw_episodes_preserved"] is True
    assert artifact["new_episode_count"] == 5
    assert artifact["heldout_trace_count"] == 0
    assert artifact["retention_score"] == pytest.approx(0.0)
    assert artifact["forgetting_rate"] == pytest.approx(1.0)
    assert artifact["adaptation_score"] == pytest.approx(1.0)
    assert artifact["consolidation_gate_passed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_learn_3304_validation_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3304-1/5/6: invalid artifacts and unsafe baselines fail closed."""

    _write_ready_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    _write_json(
        tmp_path,
        mod.EXP3291_REL_PATH,
        _baseline_payload(controller_memory_only=False),
    )
    unsafe = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    assert unsafe["fr11_redteam_repair_memory_replay_ready"] is False
    assert unsafe["heldout_trace_count"] == 0
    assert mod.baseline_safe(_baseline_payload(foundation_weight_updates_performed=True)) is False
    assert mod.collect_raw_episodes({"exp3300": {}, "exp3299": {}, "exp3302": {}}) == []
    assert mod.redteam_family_episodes({"per_probe_family_metrics": "bad"}) == []
    assert mod.selected_defense_policy_episode({"selected_defense_config_ready": True}) is None
    assert (
        mod.redteam_family_action(
            "aligned_benign",
            attack_success_rate=0.0,
            false_positive_rate=0.0,
            dataflip_gate_passed=False,
            error_count=0,
        )
        == "preserve_aligned_benign_utility_route"
    )
    assert (
        mod.redteam_family_action(
            "dataflip_kad_adaptive",
            attack_success_rate=0.0,
            false_positive_rate=0.0,
            dataflip_gate_passed=True,
            error_count=0,
        )
        == "consolidate_dataflip_guard_route"
    )
    assert (
        mod.redteam_family_action(
            "promptinject",
            attack_success_rate=0.2,
            false_positive_rate=0.0,
            dataflip_gate_passed=False,
            error_count=0,
        )
        == "route_redteam_family_to_repair:promptinject"
    )
    assert mod.controller_key_for_episode({"category": "misc", "raw_evidence": {}}) == "misc"
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert mod.read_json_object(malformed) == {}
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(list_payload) == {}
    assert mod.score_ratio(1, 0) == 0.0
    assert mod.sequence_of_mappings("bad") == []
    assert mod.path_as_artifact_string(tmp_path, Path("/outside/root/artifact.json")) == (
        "/outside/root/artifact.json"
    )
    assert mod.safe_int("bad") == 0
    assert mod.safe_float("bad") == 0.0

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="controller_memory_only"):
        mod.validate_artifact(artifact | {"controller_memory_only": False})
    with pytest.raises(ValueError, match="foundation_weight_updates_performed"):
        mod.validate_artifact(artifact | {"foundation_weight_updates_performed": True})
    with pytest.raises(ValueError, match="learned_policy_updates"):
        mod.validate_artifact(artifact | {"learned_policy_updates": []})
    with pytest.raises(ValueError, match="retention_score"):
        mod.validate_artifact(artifact | {"retention_score": 2.0})
    with pytest.raises(ValueError, match="forgetting_rate"):
        mod.validate_artifact(artifact | {"forgetting_rate": 0.5})
    with pytest.raises(ValueError, match="raw_episodes"):
        mod.validate_artifact(artifact | {"raw_episodes": "bad"})
    with pytest.raises(ValueError, match="new_episode_count"):
        mod.validate_artifact(artifact | {"new_episode_count": 99})
    with pytest.raises(ValueError, match="raw_episode_preservation_path"):
        mod.validate_artifact(artifact | {"raw_episode_preservation_path": ""})
    with pytest.raises(ValueError, match="consolidation_gate_passed"):
        mod.validate_artifact(artifact | {"consolidation_gate_passed": False})
    with pytest.raises(ValueError, match="fr11_redteam_repair_memory_replay_ready"):
        mod.validate_artifact(artifact | {"fr11_redteam_repair_memory_replay_ready": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "bad"})
