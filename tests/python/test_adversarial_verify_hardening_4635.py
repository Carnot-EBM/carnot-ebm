"""Tests for Exp 4635 adversarial_verify intrinsic-reward hardening.

Spec refs: REQ-ARC-WMTE-4635,
SCENARIO-ARC-WMTE-4635-INTRINSIC-REWARD-DOWNSTREAM,
SCENARIO-ARC-WMTE-4635-SELF-SUPERVISED-CNN-SUBSTRATE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
EXP4628 = REPO / "results" / "experiment_4628_dense_curiosity_progress_loop.json"
EXP4629 = REPO / "results" / "experiment_4629_graduate_action_effect_predictor_live.json"
INTRINSIC_KIND = "intrinsic-reward-without-downstream-gain"


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kind(report: dict[str, Any], kind: str) -> list[dict[str, str]]:
    return [flag for flag in report["flags"] if flag["kind"] == kind]


def _intrinsic_overclaim_artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4635_intrinsic_reward_overclaim_fixture",
        "game": "ar25",
        "headline": "curiosity exploration win from learning-progress bonus",
        "honest_verdict": "success: dense_curiosity_exploration_win_intrinsic_reward_up",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "intrinsic_bonus_delta": 0.37,
        "mean_intrinsic_reward_before": 0.11,
        "mean_intrinsic_reward_after": 0.48,
        "random_seed": 4635,
        "reproducibility_checksum": "sha256:" + "c" * 64,
    }
    payload.update(overrides)
    return payload


def _fast_cnn_artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4635_fast_cnn_scoring_fixture",
        "game": "cn04",
        "honest_verdict": "complete: cached_frame_change_cnn_forward_pass_scored",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "duration_s": 0.23,
        "model_specs": {
            "architecture": "self-supervised CNN action-effect frame-change predictor",
            "framework": "torch",
            "input": "cached ARC frames",
        },
        "cnn_substrate": "cached_frame_change_cnn_forward_pass",
        "random_seed": 4635,
        "reproducibility_checksum": "sha256:" + "d" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4635_spec_declares_intrinsic_reward_and_cnn_contract() -> None:
    """REQ-ARC-WMTE-4635: OpenSpec declares both adversarial_verify hardenings."""

    from carnot import experiment_4635_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4635" in spec
    assert "SCENARIO-ARC-WMTE-4635-INTRINSIC-REWARD-DOWNSTREAM" in spec
    assert "SCENARIO-ARC-WMTE-4635-SELF-SUPERVISED-CNN-SUBSTRATE" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4635_exploration_win_claim_needs_downstream_metric(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4635-INTRINSIC-REWARD-DOWNSTREAM: bonus-only wins warn."""

    report = _report_for_payload(tmp_path, _intrinsic_overclaim_artifact())
    flags = _flag_kind(report, INTRINSIC_KIND)

    assert flags
    assert flags[0]["severity"] == "warn"
    assert "intrinsic-bonus" in flags[0]["detail"]
    assert "downstream" in flags[0]["detail"]


def test_scenario_arc_wmte_4635_honest_intrinsic_bonus_diagnostic_not_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4635-INTRINSIC-REWARD-DOWNSTREAM: diagnostic only is clean."""

    report = _report_for_payload(
        tmp_path,
        _intrinsic_overclaim_artifact(
            headline="diagnostic: curiosity bonus magnitude increased during replay",
            honest_verdict="complete: dense_curiosity_bonus_diagnostic_only_no_win_claim",
        ),
    )

    assert _flag_kind(report, INTRINSIC_KIND) == []


def test_scenario_arc_wmte_4635_downstream_metric_exemplar_not_false_flagged() -> None:
    """REQ-ARC-WMTE-4635: exp4628 has downstream deltas and should stay clean."""

    report = av.verify_artifact(EXP4628)

    assert _flag_kind(report, INTRINSIC_KIND) == []


def test_scenario_arc_wmte_4635_a1_fixture_has_no_adversarial_flags() -> None:
    """REQ-ARC-WMTE-4635: exp4628 downstream zero deltas are honest-null evidence."""

    report = av.verify_artifact(EXP4628)

    assert report["flags"] == []


def test_scenario_arc_wmte_4635_methodology_fast_cnn_not_duration_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4635-SELF-SUPERVISED-CNN-SUBSTRATE: method-bearing CNN passes."""

    artifact = _fast_cnn_artifact()
    report = _report_for_payload(tmp_path, artifact)
    floor = av.duration_floor_for_artifact(artifact)

    assert floor is not None
    assert floor["reason"] == "cheap_learned_value_scoring"
    assert floor["marker"] == "cnn"
    assert float(floor["min_duration_s"]) <= artifact["duration_s"]
    assert _flag_kind(report, "DURATION_TOO_SHORT") == []


def test_scenario_arc_wmte_4635_no_methodology_fast_cnn_still_duration_flags(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4635-SELF-SUPERVISED-CNN-SUBSTRATE: missing methodology fires."""

    artifact = _fast_cnn_artifact()
    artifact.pop("model_specs")
    artifact.pop("random_seed")
    artifact.pop("reproducibility_checksum")
    artifact["methodology_note"] = "torch CNN cached frame-change scoring claimed without fields"
    report = _report_for_payload(tmp_path, artifact)
    flags = _flag_kind(report, "DURATION_TOO_SHORT")

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "verifier-scoring" in flags[0]["detail"]


def test_scenario_arc_wmte_4635_a2_fixture_not_duration_flagged() -> None:
    """REQ-ARC-WMTE-4635: exp4629 remains clean for DURATION_TOO_SHORT."""

    report = av.verify_artifact(EXP4629)

    assert _flag_kind(report, "DURATION_TOO_SHORT") == []


def test_req_arc_wmte_4635_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4635: Exp 4635 emits the required evidence fields."""

    from carnot import experiment_4635_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4635": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: adversarial_verify_hardened_intrinsic_reward_guard_plus_cnn_substrate_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["intrinsic_reward_overclaim_guard_added"] is True
    assert artifact["cnn_substrate_floor_added"] is True
    assert artifact["honest_diagnostic_not_flagged"] is True
    assert artifact["no_methodology_fast_run_still_fires"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["research_conductor_modified"] is False
    assert artifact["random_seed"] == 4635
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4635_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-4635: artifact validation fails closed."""

    from carnot import experiment_4635_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4635": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["intrinsic_reward_overclaim_guard_added"] = False
    bad["cnn_substrate_floor_added"] = False
    bad["honest_diagnostic_not_flagged"] = False
    bad["no_methodology_fast_run_still_fires"] = False
    bad["tests_added"] = {"passed": False}
    bad["research_conductor_modified"] = True
    bad["random_seed"] = 0
    bad["preconditions_checked"] = {"ok": False}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "intrinsic_reward_overclaim_guard_added" in errors
    assert "cnn_substrate_floor_added" in errors
    assert "honest_diagnostic_not_flagged" in errors
    assert "no_methodology_fast_run_still_fires" in errors
    assert "tests_added.passed" in errors
    assert "research_conductor_modified" in errors
    assert "random_seed" in errors
    assert "preconditions_checked.ok" in errors
    assert "field_principles.honest_verdict" in errors
    assert "reproducibility_checksum" in errors

    bad_types = dict(artifact)
    bad_types["tests_added"] = None
    bad_types["preconditions_checked"] = None
    bad_types["field_principles"] = None
    type_errors = mod.validate_artifact(bad_types)

    assert "tests_added" in type_errors
    assert "preconditions_checked" in type_errors
    assert "field_principles" in type_errors
