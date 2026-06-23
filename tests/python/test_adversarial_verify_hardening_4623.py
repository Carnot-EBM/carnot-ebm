"""Tests for Exp 4623 adversarial_verify bridge hardening.

Spec refs: REQ-ARC-WMTE-4623,
SCENARIO-ARC-WMTE-4623-OFFLINE-LIVE-OVERCLAIM,
SCENARIO-ARC-WMTE-4623-CHEAP-VALUE-SUBSTRATE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
EXP4617 = REPO / "results" / "experiment_4617_graduate_spatial_value_head_live.json"


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kind(report: dict[str, Any], kind: str) -> list[dict[str, str]]:
    return [flag for flag in report["flags"] if flag["kind"] == kind]


def _live_overclaim_artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4623_arc_live_overclaim_fixture",
        "game": "ar25",
        "honest_verdict": "success: live_agent_first_win_efficiency_up_from_value_head",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "offline_loo_auroc": 0.725,
        "offline_detector_auroc": 0.725,
        "random_seed": 4623,
        "reproducibility_checksum": "sha256:" + "a" * 64,
    }
    payload.update(overrides)
    return payload


def _fast_value_head_artifact(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4623_fast_value_head_fixture",
        "game": "cn04",
        "honest_verdict": "complete: cached_value_head_forward_pass_scored",
        "inference_substrate": av.VERIFIER_SCORING_SUBSTRATE,
        "duration_s": 0.44,
        "model_specs": {
            "architecture": "linear value-head forward pass",
            "framework": "torch",
            "input": "cached ARC candidate states",
        },
        "value_head_substrate": "cached_candidate_linear_forward_pass",
        "random_seed": 4623,
        "reproducibility_checksum": "sha256:" + "b" * 64,
    }
    payload.update(overrides)
    return payload


def test_req_arc_wmte_4623_spec_declares_reader_hardening_contract() -> None:
    """REQ-ARC-WMTE-4623: OpenSpec declares both adversarial_verify guards."""

    from carnot import experiment_4623_adversarial_verify_hardening as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4623" in spec
    assert "SCENARIO-ARC-WMTE-4623-OFFLINE-LIVE-OVERCLAIM" in spec
    assert "SCENARIO-ARC-WMTE-4623-CHEAP-VALUE-SUBSTRATE" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4623_live_win_claim_needs_live_metric(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4623-OFFLINE-LIVE-OVERCLAIM: offline AUROC is not a live win."""

    report = _report_for_payload(tmp_path, _live_overclaim_artifact())
    flags = _flag_kind(report, "OFFLINE_SUBSTITUTED_FOR_LIVE")

    assert flags
    assert flags[0]["severity"] == "warn"
    assert "offline AUROC" in flags[0]["detail"]
    assert "live metric" in flags[0]["detail"]


def test_scenario_arc_wmte_4623_honest_offline_auroc_result_not_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4623-OFFLINE-LIVE-OVERCLAIM: offline-only results pass."""

    report = _report_for_payload(
        tmp_path,
        _live_overclaim_artifact(
            honest_verdict="complete: offline_auroc_characterized_bridge_gap_open"
        ),
    )

    assert _flag_kind(report, "OFFLINE_SUBSTITUTED_FOR_LIVE") == []


def test_scenario_arc_wmte_4623_live_metric_exemplar_not_false_flagged() -> None:
    """REQ-ARC-WMTE-4623: exp4617 has live metrics and should not trip guard 1."""

    report = av.verify_artifact(EXP4617)

    assert _flag_kind(report, "OFFLINE_SUBSTITUTED_FOR_LIVE") == []


def test_scenario_arc_wmte_4623_methodology_fast_value_head_not_duration_flagged(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4623-CHEAP-VALUE-SUBSTRATE: method-bearing 0.44s pass."""

    artifact = _fast_value_head_artifact()
    report = _report_for_payload(tmp_path, artifact)
    floor = av.duration_floor_for_artifact(artifact)

    assert floor is not None
    assert floor["reason"] == "cheap_learned_value_scoring"
    assert float(floor["min_duration_s"]) <= artifact["duration_s"]
    assert _flag_kind(report, "DURATION_TOO_SHORT") == []


def test_scenario_arc_wmte_4623_no_methodology_fast_run_still_duration_flags(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4623-CHEAP-VALUE-SUBSTRATE: missing methodology still fires."""

    artifact = _fast_value_head_artifact()
    artifact.pop("model_specs")
    artifact.pop("random_seed")
    artifact.pop("reproducibility_checksum")
    artifact["methodology_note"] = "torch cached value-head scoring claimed without fields"
    report = _report_for_payload(tmp_path, artifact)
    flags = _flag_kind(report, "DURATION_TOO_SHORT")

    assert flags
    assert flags[0]["severity"] == "critical"
    assert "verifier-scoring" in flags[0]["detail"]


def test_req_arc_wmte_4623_runner_builds_required_terminal_artifact() -> None:
    """REQ-ARC-WMTE-4623: Exp 4623 emits the required evidence fields."""

    from carnot import experiment_4623_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4623": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: adversarial_verify_hardened_offline_live_overclaim_guard_plus_cheap_value_substrate_tests_green."
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["offline_live_overclaim_guard_added"] is True
    assert artifact["cheap_value_substrate_floor_added"] is True
    assert artifact["honest_offline_result_not_flagged"] is True
    assert artifact["no_methodology_fast_run_still_fires"] is True
    assert artifact["tests_added"]["passed"] is True
    assert artifact["research_conductor_modified"] is False
    assert artifact["random_seed"] == 4623
    assert artifact["preconditions_checked"]["ok"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4623_runner_validation_rejects_malformed_artifact() -> None:
    """REQ-ARC-WMTE-4623: artifact validation fails closed."""

    from carnot import experiment_4623_adversarial_verify_hardening as mod

    artifact = mod.build_artifact(
        root=REPO,
        preconditions_checked={
            "agents_md_read": True,
            "codex_or_opencode_md_read": True,
            "adversarial_verify_parse_ok": True,
            "fixtures_present": True,
            "spec_has_req_4623": True,
            "research_conductor_modified": False,
            "network_required": False,
            "ok": True,
        },
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["offline_live_overclaim_guard_added"] = False
    bad["cheap_value_substrate_floor_added"] = False
    bad["honest_offline_result_not_flagged"] = False
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
    assert "offline_live_overclaim_guard_added" in errors
    assert "cheap_value_substrate_floor_added" in errors
    assert "honest_offline_result_not_flagged" in errors
    assert "no_methodology_fast_run_still_fires" in errors
    assert "tests_added.passed" in errors
    assert "research_conductor_modified" in errors
    assert "random_seed" in errors
    assert "preconditions_checked.ok" in errors
    assert "field_principles.honest_verdict" in errors
    assert "reproducibility_checksum" in errors
