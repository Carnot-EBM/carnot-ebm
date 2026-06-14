"""Tests for Exp 4195 .388 capstone aggregation.

Spec refs: REQ-CAPSTONE-4195, SCENARIO-CAPSTONE-4195.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v388_4195 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[int, JsonDict]:
    return {
        4185: {
            "honest_verdict": "complete: headroom fixture",
            "acceptance_gate": True,
            "headroom_present_domain": "code",
            "max_selectable_headroom": 0.18,
            "per_domain_headroom": {"code": {"selectable_headroom": 0.18}},
        },
        4186: {
            "honest_verdict": "complete: efficiency fixture",
            "verifier_efficiency_win": True,
            "positive_control_confirmed": True,
            "accuracy_parity_vs_judge": {
                "arm_a_pass1": 0.84,
                "arm_j_pass1": 0.66,
                "delta": 0.18,
                "ci95": [0.08, 0.3],
                "within_ci_or_better": True,
            },
            "cost_ratio_vs_judge": {
                "ten_x_cheaper_on_both_axes": True,
                "strictly_pareto_dominant": True,
                "wall_clock_x_cheaper": 500000.0,
                "arm_a_total_tokens": 0,
                "arm_j_total_tokens": 5270,
            },
        },
        4187: {
            "honest_verdict": "complete: gap4 safety fixture",
            "graded_gate_pass2_vs_vote": 0.129,
            "gross_recovery_ledger": {"recovered": 4, "lost": 0},
            "pass2_vote_wins_lost": 0,
            "vote_aware_guard_blocked_mispromotion": True,
            "vote_aware_guard": {"blocked_tasks": ["25094a63"]},
        },
        4188: {
            "honest_verdict": "success: sovereign fixture",
            "no_closed_weight_call": True,
            "self_distillation_corpus_size": 7,
            "sovereign_pool_pass2": {
                "LOCAL_HARDENED_GATE": 0.4839,
                "TRM_VOTE": 0.4516,
                "delta_vs_vote": 0.0323,
                "recovered": 1,
                "lost": 0,
            },
        },
        4189: {
            "honest_verdict": "blocked_diffusiongemma_not_cached",
            "acceptance_gate": True,
            "diffusiongemma_feasible": False,
            "guided_vs_unguided_delta": {"status": "blocked_diffusiongemma_not_cached"},
        },
        4190: {
            "honest_verdict": "success: arc fixture",
            "acceptance_gate_passed": True,
            "total_levels_solved": 15,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 1,
            "real_env_confirmed": True,
        },
        4191: {
            "honest_verdict": "complete: live env fixture",
            "acceptance_gate_passed": True,
            "live_env_reachable": True,
            "environment_count": 25,
            "random_greedy_baseline": {"actions_vs_baseline_actions": 0.35},
        },
        4192: {
            "honest_verdict": "complete: sota fixture",
            "flagged_for_v389": "s3_diffusiongemma_verifier_guided_search_scaleup_v389",
        },
        4193: {
            "honest_verdict": "complete: registry fixture",
            "regression_guard_passed": True,
        },
        4194: {
            "honest_verdict": "complete: hardware fixture",
            "kv260_reachable": True,
            "gatemate_reachable": False,
            "polarfire_reachable": True,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4195_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4195: OpenSpec declares the .388 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4195" in spec
    assert "SCENARIO-CAPSTONE-4195" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for status in mod.EFFICIENCY_MOAT_STATUSES:
        assert status in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "flagged_adversarial:true" in spec
    assert "verifier_efficiency_win:true" in spec
    assert "vote_aware_guard_blocked_mispromotion:true" in spec


def test_scenario_capstone_4195_current_artifacts_are_decision_grade() -> None:
    """SCENARIO-CAPSTONE-4195: current clean .388 artifacts aggregate honestly."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=10.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "efficiency_moat_won"
    assert artifact["honest_verdict"].startswith("complete: capstone_v388_efficiency_moat_won_")
    assert artifact["efficiency_moat_status"] == "WON"
    assert artifact["efficiency_moat"]["verifier_efficiency_win"] is True
    assert artifact["efficiency_moat"]["positive_control_confirmed"] is True
    assert artifact["gap4_production_safe"] is True
    assert artifact["gap4_production_safety"]["gross_recovery_ledger"] == {
        "lost": 0,
        "recovered": 4,
    }
    assert artifact["gap4_sovereign"] is False
    assert artifact["gap4_sovereign_detail"]["status"] == "skipped_flagged_adversarial"
    assert artifact["gap4_sovereign_detail"]["self_distillation_corpus_size"] is None
    assert artifact["diffusiongemma_feasible"] is False
    assert artifact["diffusiongemma_detail"]["status"] == "blocked_diffusiongemma_not_cached"
    assert artifact["total_arc_levels_solved"] == 15
    assert artifact["live_env_reachable"] is True
    assert artifact["strongest_sota_flagged_for_v389"] == (
        "s3_diffusiongemma_verifier_guided_search_scaleup_v389"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    skipped_ids = {row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]}
    assert skipped_ids == {4188, 4193}
    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    for experiment_id, rel_path in mod.DEFAULT_UPSTREAM_PATHS.items():
        expected_sha = hashlib.sha256(rel_path.read_bytes()).hexdigest()
        assert provenance[experiment_id]["sha256"] == expected_sha
    assert provenance[4188]["skipped"] is True
    assert provenance[4188]["fields_imported"] == []
    assert provenance[4193]["skipped"] is True
    assert provenance[4193]["fields_imported"] == []
    assert "verifier_efficiency_win" in provenance[4186]["fields_imported"]
    assert "gross_recovery_ledger" in provenance[4187]["fields_imported"]
    assert "total_levels_solved" in provenance[4190]["fields_imported"]
    assert "live_env_reachable" in provenance[4191]["fields_imported"]
    assert "flagged_for_v389" in provenance[4192]["fields_imported"]


def test_req_capstone_4195_efficiency_and_sovereign_branching(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4195: headline branches distinguish win, bounded, judge, and sovereign."""

    payloads = _minimal_payloads()
    payloads[4186]["verifier_efficiency_win"] = False
    payloads[4186]["cost_ratio_vs_judge"]["ten_x_cheaper_on_both_axes"] = False
    payloads[4186]["cost_ratio_vs_judge"]["strictly_pareto_dominant"] = False
    _write_default_artifacts(tmp_path, payloads)
    sovereign = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)
    mod.validate_artifact(sovereign)
    assert sovereign["efficiency_moat_status"] == "BOUNDED-no-cost-advantage"
    assert sovereign["headline_outcome"] == "gap4_production_safe_and_sovereign"
    assert sovereign["gap4_sovereign"] is True
    assert sovereign["gap4_sovereign_detail"]["self_distillation_corpus_size"] == 7

    payloads = _minimal_payloads()
    payloads[4186]["verifier_efficiency_win"] = False
    payloads[4186]["accuracy_parity_vs_judge"]["arm_a_pass1"] = 0.60
    payloads[4186]["accuracy_parity_vs_judge"]["arm_j_pass1"] = 0.70
    _write_default_artifacts(tmp_path, payloads)
    judge = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert judge["efficiency_moat_status"] == "JUDGE-MORE-ACCURATE"
    assert judge["headline_outcome"] == "efficiency_moat_judge_more_accurate"

    payloads = _minimal_payloads()
    payloads[4186]["verifier_efficiency_win"] = False
    payloads[4188]["sovereign_pool_pass2"]["delta_vs_vote"] = 0.0
    payloads[4188]["self_distillation_corpus_size"] = 0
    _write_default_artifacts(tmp_path, payloads)
    under = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert under["gap4_sovereign"] is False
    assert under["headline_outcome"] == "gap4_sovereign_under_induces"

    payloads = _minimal_payloads()
    payloads[4186]["verifier_efficiency_win"] = False
    payloads[4187]["gross_recovery_ledger"] = {"recovered": 3, "lost": 1}
    payloads[4187]["vote_aware_guard_blocked_mispromotion"] = False
    _write_default_artifacts(tmp_path, payloads)
    bounded = mod.build_artifact(tmp_path, started_s=3.5, now_s=3.75)
    assert bounded["gap4_production_safe"] is False
    assert bounded["headline_outcome"] == "efficiency_moat_bounded_no_cost_advantage"

    payloads = _minimal_payloads()
    payloads[4186]["flagged_adversarial"] = True
    payloads[4188]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)
    flagged = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert flagged["efficiency_moat"]["status"] == "skipped_flagged_adversarial"
    assert flagged["upstream_provenance"][1]["experiment_id"] == 4186
    assert flagged["upstream_provenance"][1]["fields_imported"] == []
    assert flagged["flagged_artifacts_skipped"][0]["experiment_id"] == 4186


def test_req_capstone_4195_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4195: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4195_capstone_v388.json"),
        started_s=6.0,
        now_s=6.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_4194 = _minimal_payloads()
    payloads_without_4194.pop(4194)
    _write_default_artifacts(missing_root, payloads_without_4194)
    missing = mod.build_artifact(missing_root, started_s=7.0, now_s=7.25)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4194}]
    assert missing["hardware_continuity"]["status"] == "missing"

    assert mod.efficiency_moat({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.efficiency_moat(None, was_skipped=False)["status"] == "missing"
    assert mod.gap4_production_safety({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.gap4_production_safety(None, was_skipped=False)["status"] == "missing"
    assert mod.gap4_sovereignty({}, was_skipped=True)["status"] == ("skipped_flagged_adversarial")
    assert mod.gap4_sovereignty(None, was_skipped=False)["status"] == "missing"
    assert mod.diffusiongemma({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.diffusiongemma(None, was_skipped=False)["status"] == "missing"
    assert mod.arc_progress({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_progress(None, was_skipped=False)["status"] == "missing"
    assert mod.live_env({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.live_env(None, was_skipped=False)["status"] == "missing"
    assert mod.sota_v389({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.sota_v389(None, was_skipped=False)["status"] == "missing"
    assert mod.registry_hygiene({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_hygiene(None, was_skipped=False)["status"] == "missing"
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.hardware_continuity(None, was_skipped=False)["status"] == "missing"
    assert mod.headroom_harness({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.headroom_harness(None, was_skipped=False)["status"] == "missing"

    expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    expect_validation_error(artifact, lambda a: a.update({"headline_outcome": "bad"}), "headline")
    expect_validation_error(
        artifact, lambda a: a.update({"efficiency_moat_status": "bad"}), "efficiency"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"gap4_production_safe": "true"}), "gap4_production"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"gap4_sovereign": "false"}), "gap4_sovereign"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"diffusiongemma_feasible": "false"}), "diffusion"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"total_arc_levels_solved": 13}), "ARC levels"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"inference_substrate": "bad"}), "inference_substrate"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"field_principles": []}), "field_principles"
    )
    expect_validation_error(
        artifact,
        lambda a: a["field_principles"].update({"honest_verdict": "wrong"}),
        "principle",
    )
    expect_validation_error(
        artifact, lambda a: a.update({"upstream_provenance": {}}), "upstream_provenance"
    )
    expect_validation_error(
        artifact, lambda a: a["upstream_provenance"].append(42), "entries must be objects"
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4185"}),
        "integer experiment_id",
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"fields_imported": "bad"}),
        "fields_imported",
    )
    expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][3].update(
            {"skipped": True, "fields_imported": ["self_distillation_corpus_size"]}
        ),
        "skipped upstreams",
    )
    expect_validation_error(
        artifact, lambda a: a["upstream_provenance"][0].update({"sha256": "bad"}), "sha256"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"flagged_artifacts_skipped": {}}), "flagged"
    )
    expect_validation_error(
        artifact, lambda a: a.update({"reproducibility_checksum": "bad"}), "checksum"
    )
    expect_validation_error(artifact, lambda a: a.update({"duration_s": 99.0}), "checksum")

    output_path = tmp_path / "results" / "experiment_4195_capstone_v388.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out


def expect_validation_error(artifact: JsonDict, mutator: Any, match: str) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)
