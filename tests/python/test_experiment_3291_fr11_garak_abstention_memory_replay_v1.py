"""Tests for Exp 3291 FR-11 Garak and abstention memory replay.

Spec refs: REQ-LEARN-3291, SCENARIO-LEARN-3291,
SCENARIO-LEARN-3291-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_garak_abstention_memory_replay_v1 as mod


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
        "adaptation_score": 1.0,
        "artifact": "experiment_3278_fr11_full_corpus_continual_self_learning_audit_v1",
        "before_after_metrics": {
            "heldout_trace_count": 2056,
            "legacy_heldout_count": 56,
            "legacy_retained_count": 55,
            "negative_transfer_rate": 0.0,
            "retention_score": 0.982143,
        },
        "continuous_self_learning_task": True,
        "controller_memory_only": controller_memory_only,
        "foundation_weight_updates_performed": foundation_weight_updates_performed,
        "fr11_full_corpus_audit_ready": True,
        "heldout_trace_count": 2056,
        "honest_verdict": "complete: baseline controller-memory only",
        "negative_transfer_rate": 0.0,
        "retention_score": 0.982143,
    }


def _exp3282_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp3282",
        "garak_install_probe_manifest_ready": True,
        "garak_runner_ready": True,
        "garak_available": True,
        "install_blockers": [],
        "preconditions_checked": [
            {
                "name": "project_garak_import",
                "passed": False,
                "blocked_reason": "blocked_garak_import_unavailable",
            },
            {
                "name": "isolated_uv_garak_import",
                "passed": True,
                "blocked_reason": "",
                "version": "0.15.0",
            },
        ],
        "honest_verdict": "complete: garak runner available through isolated uv",
    }


def _exp3285_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp3285",
        "garak_dataflip_redteam_eval_v2_ready": True,
        "garak_redteam_eval_ready": True,
        "garak_gate_passed": False,
        "garak_probe_count": 90,
        "attack_success_rate": 0.311111,
        "blocked_reasons": ["garak_attack_success_or_error_gate_failed"],
        "honest_verdict": "complete: garak evidence ready; gate failed",
    }


def _exp3286_payload() -> dict[str, Any]:
    return {
        "experiment_id": "exp3286",
        "abstention_root_cause_audit_ready": True,
        "abstention_root_cause_identified": True,
        "dominant_root_cause": "model_output_parser_contract_mismatch",
        "prior_abstention_rate": 1.0,
        "answerable_row_count": 6,
        "abstention_reason_counts": {"model_output_unparseable": 6},
        "calibrated_rerun_plan": {"experiment_id": "exp3287"},
        "honest_verdict": "complete: clean verifier abstention root cause identified",
    }


def _exp3289_payload(*, repair_gate_open: bool = True) -> dict[str, Any]:
    return {
        "experiment_id": "exp3289",
        "repair_gate_decision_v9_ready": True,
        "repair_gate_open": repair_gate_open,
        "garak_redteam_eval_ready": True,
        "clean_verifier_rerun_ready": True,
        "kan_boundary_decision_ready": True,
        "blocked_reasons": [] if repair_gate_open else ["clean_verifier_rerun_not_ready"],
        "gate_inputs": {
            "exp3288_kan_boundary": {
                "kan_boundary_decision_ready": True,
                "kan_boundary_decision": "retire_from_prompt_injection_headline",
                "kan_downstream_use_bounded": True,
                "prior_full_corpus_auroc": 0.475326,
            }
        },
        "permitted_repair_scope": {
            "repair_task_id": "exp3290-gated-sota-repair-micro-panel-v10",
            "repair_generation_allowed": repair_gate_open,
            "scope_label": "bounded_exact_fixture_code_repair_micro_panel",
        },
        "honest_verdict": "complete: repair gate evaluated",
    }


def _write_ready_sources(tmp_path: Path) -> None:
    _write_json(tmp_path, mod.EXP3278_REL_PATH, _baseline_payload())
    _write_json(tmp_path, mod.EXP3282_REL_PATH, _exp3282_payload())
    _write_json(tmp_path, mod.EXP3285_REL_PATH, _exp3285_payload())
    _write_json(tmp_path, mod.EXP3286_REL_PATH, _exp3286_payload())
    _write_json(tmp_path, mod.EXP3289_REL_PATH, _exp3289_payload())


def test_req_learn_3291_spec_anchor_declares_replay_schema() -> None:
    """REQ-LEARN-3291: OpenSpec declares the .304 replay contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3291" in spec
    assert "SCENARIO-LEARN-3291" in spec
    assert "SCENARIO-LEARN-3291-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_3291_collects_raw_blocker_episodes(tmp_path: Path) -> None:
    """REQ-LEARN-3291-2/3/4: .304 evidence becomes raw controller episodes."""

    _write_ready_sources(tmp_path)
    sources = mod.load_sources(tmp_path)
    episodes = mod.collect_raw_episodes(sources)
    memory = mod.build_controller_memory(episodes)
    replay = mod.replay_new_episodes(memory, episodes)

    assert [row["category"] for row in episodes] == [
        "garak_toolchain",
        "garak_redteam",
        "clean_verifier_abstention",
        "kan_boundary",
        "repair_gate",
    ]
    assert all(row["raw_evidence"] for row in episodes)
    assert memory["controller_memory_only"] is True
    assert memory["consolidation_allowed"] is True
    assert replay["new_episode_count"] == 5
    assert replay["adapted_episode_count"] == 5
    assert replay["adaptation_score"] == pytest.approx(1.0)
    assert mod.blocked_trace_categories(episodes) == [
        "clean_verifier_abstention",
        "garak_redteam",
        "garak_toolchain",
        "kan_boundary",
    ]


def test_scenario_learn_3291_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3291: ready sources write controller-memory replay output."""

    _write_ready_sources(tmp_path)
    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        monotonic=iter([10.0, 12.5]).__next__,
        random_seed=3291,
        tests_run=["SCENARIO-LEARN-3291"],
    )
    written = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["fr11_garak_abstention_memory_replay_ready"] is True
    assert artifact["controller_memory_only"] is True
    assert artifact["foundation_weight_updates_performed"] is False
    assert artifact["raw_episodes_preserved"] is True
    assert artifact["new_episode_count"] == 5
    assert artifact["heldout_trace_count"] == 2056
    assert artifact["retention_score"] == pytest.approx(0.982143)
    assert artifact["adaptation_score"] == pytest.approx(1.0)
    assert artifact["forgetting_rate"] == pytest.approx(0.017857)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["blocked_trace_categories"] == [
        "clean_verifier_abstention",
        "garak_redteam",
        "garak_toolchain",
        "kan_boundary",
    ]
    assert artifact["memory_update_policy"]["foundation_weight_updates_allowed"] is False
    assert artifact["memory_update_policy"]["raw_episode_preservation_required"] is True
    assert artifact["raw_episodes"] == mod.collect_raw_episodes(mod.load_sources(tmp_path))
    assert artifact["tests_run"] == ["SCENARIO-LEARN-3291"]
    assert artifact["random_seed"] == 3291
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert "foundation_weight_updates_performed=false" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_scenario_learn_3291_missing_baseline_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3291-BLOCKED: .304 episodes survive missing baseline."""

    _write_json(tmp_path, mod.EXP3282_REL_PATH, _exp3282_payload())
    _write_json(tmp_path, mod.EXP3285_REL_PATH, _exp3285_payload())
    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        monotonic=iter([1.0, 1.25]).__next__,
    )

    assert artifact["fr11_garak_abstention_memory_replay_ready"] is False
    assert artifact["blocked_reason"] == "baseline_exp3278_missing_or_unsafe"
    assert artifact["controller_memory_only"] is True
    assert artifact["foundation_weight_updates_performed"] is False
    assert artifact["raw_episodes_preserved"] is True
    assert artifact["new_episode_count"] == 5
    assert artifact["heldout_trace_count"] == 0
    assert artifact["retention_score"] == pytest.approx(0.0)
    assert artifact["forgetting_rate"] == pytest.approx(1.0)
    assert artifact["adaptation_score"] == pytest.approx(1.0)
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_learn_3291_validation_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3291-1/5/6: unsafe baseline and invalid artifacts fail closed."""

    _write_ready_sources(tmp_path)
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    _write_json(
        tmp_path,
        mod.EXP3278_REL_PATH,
        _baseline_payload(controller_memory_only=False),
    )
    unsafe = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    assert unsafe["fr11_garak_abstention_memory_replay_ready"] is False
    assert unsafe["heldout_trace_count"] == 0
    assert mod.garak_toolchain_episode({})["observed_signal"] == "missing_artifact"
    assert mod.garak_redteam_episode({})["observed_signal"] == "missing_artifact"
    assert mod.kan_boundary_episode({"gate_inputs": {}})["observed_signal"] == "missing_artifact"
    assert mod.baseline_safe(_baseline_payload(foundation_weight_updates_performed=True)) is False
    assert mod.score_ratio(1, 0) == 0.0
    assert mod.sequence_of_mappings("bad") == []
    assert mod.sequence_values("bad") == []
    assert mod.path_as_artifact_string(tmp_path, Path("/outside/root/artifact.json")) == (
        "/outside/root/artifact.json"
    )
    assert mod.first_nonempty(["", None]) == "no_signal"
    assert mod.safe_int("bad") == 0
    assert mod.safe_float("bad") == 0.0

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="controller_memory_only"):
        mod.validate_artifact(artifact | {"controller_memory_only": False})
    with pytest.raises(ValueError, match="foundation_weight_updates_performed"):
        mod.validate_artifact(artifact | {"foundation_weight_updates_performed": True})
    with pytest.raises(ValueError, match="raw_episodes_preserved"):
        mod.validate_artifact(artifact | {"raw_episodes_preserved": False})
    with pytest.raises(ValueError, match="retention_score"):
        mod.validate_artifact(artifact | {"retention_score": 2.0})
    with pytest.raises(ValueError, match="forgetting_rate"):
        mod.validate_artifact(artifact | {"forgetting_rate": 0.5})
    with pytest.raises(ValueError, match="raw_episodes"):
        mod.validate_artifact(artifact | {"raw_episodes": "bad"})
    with pytest.raises(ValueError, match="new_episode_count"):
        mod.validate_artifact(artifact | {"new_episode_count": 99})
    with pytest.raises(ValueError, match="memory_update_policy"):
        mod.validate_artifact(artifact | {"memory_update_policy": {}})
    with pytest.raises(ValueError, match="fr11_garak_abstention_memory_replay_ready"):
        mod.validate_artifact(artifact | {"fr11_garak_abstention_memory_replay_ready": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "done"})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "bad"})
