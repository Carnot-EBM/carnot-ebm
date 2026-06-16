"""Tests for Exp 4301 .397 capstone aggregation.

Spec refs: REQ-CAPSTONE-4301, SCENARIO-CAPSTONE-4301.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v397_4301 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _paper_gate(paper_ready: bool = True) -> JsonDict:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
        "note": "fixture publication gate",
    }


def _minimal_payloads() -> dict[str, JsonDict]:
    return {
        "4291_cross_generator": {
            "honest_verdict": "complete: cross-generator fixture",
            "cross_generator_holds": True,
            "cross_generator_delta": 0.32,
            "cross_generator_ci95": [0.08, 0.55],
            "vote_at_1": 0.2,
            "oracle_at_k": 0.72,
            "pass_rates": {"vote_at_1": 0.2, "set_encoder_at_1": 0.52},
            "non_degenerate_guards_pass": True,
            "per_substrate_delta": {"arcgen": {"cross_generator_delta": 0.32}},
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4292_partial_state": {
            "honest_verdict": "complete: partial-state scorer fixture",
            "partial_state_scorer_built": True,
            "partial_state_leak_free": True,
            "partial_state_auroc": 0.81,
            "leak_ablation_auroc": 0.77,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4293_generation": {
            "honest_verdict": "complete: in-generation moat fixture",
            "diffusiongemma_guidance_moat": True,
            "carnot_minus_rfg_delta": 0.21,
            "guidance_moat_ci95": [0.04, 0.37],
            "guidance_changes_selection": True,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4294_efficiency": {
            "honest_verdict": "complete: efficiency fixture",
            "efficiency_pareto_holds": True,
            "cost_ratio": 0.04,
            "energy_accuracy": 0.68,
            "best_prompted_judge_accuracy": 0.67,
            "accuracy_delta_ci95": [-0.01, 0.05],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4295_self_learning": {
            "honest_verdict": "complete: self-learning fixture",
            "online_adaptation_helps": True,
            "static_cross_family_delta": 0.2,
            "online_cross_family_delta": 0.33,
            "online_minus_static_ci95": [0.02, 0.22],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4296_arc_progress": {
            "honest_verdict": "success: arc progress fixture",
            "total_levels": 22,
            "levels_completed": 1,
            "new_levels_solved_this_task": 1,
            "game_advanced": "fixture-game",
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4299_registry": {
            "honest_verdict": "complete: registry fixture",
            "registry_reconciled": True,
            "regression_guard_passed": True,
            "gaps_logged": [],
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
        "4300_hardware": {
            "honest_verdict": "complete: hardware fixture",
            "kv260_terminal_confirmed": False,
            "kv260_step_taken": "kv260_fixture_step",
            "polarfire_step_taken": "polarfire_fixture_step",
            "gatemate_step_taken": "gatemate_fixture_step",
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4301_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4301: OpenSpec declares the .397 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4301" in spec
    assert "SCENARIO-CAPSTONE-4301" in spec
    assert "experiment_4301_capstone_v397.json" in spec
    assert "blocked_v397_artifacts_missing" in spec
    assert "publication_gate.py --json" in spec
    assert "flagged_adversarial:true" in spec
    assert "verifier_is_oracle:false" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4301_current_artifacts_block_on_missing_core() -> None:
    """SCENARIO-CAPSTONE-4301: missing exp4294 produces the honest blocked verdict."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v397_artifacts_missing"
    assert artifact["headline_outcome"] == "blocked_v397_artifacts_missing"
    assert artifact["cross_generator_moat_closes"] is False
    assert artifact["in_generation_moat_holds"] is False
    assert artifact["efficiency_pareto_hardened"] is False
    assert artifact["verifier_thesis_state"] == "partial_state_blocked"
    assert artifact["paper_ready"] is None
    assert artifact["missing_upstream_artifacts"] == [
        {"artifact_key": "4294_efficiency", "experiment_id": 4294, "reason": "missing"}
    ]
    assert artifact["reproducibility_checksum"] == mod.BLOCKED_CHECKSUM


def test_req_capstone_4301_clean_fixture_reports_scorecard_bools(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4301: clean oracle-distinct inputs can headline the scorecard."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: v397_cross_generator_closed_in_generation_moat_"
        "efficiency_hardened_arc22"
    )
    assert artifact["headline_outcome"] == (
        "in_generation_moat_holds_cross_generator_closed_partial_state_leak_free_"
        "efficiency_hardened_self_learning_helps_arc22_game_fixture-game_paper_not_ready"
    )
    assert artifact["cross_generator_moat_closes"] is True
    assert artifact["in_generation_moat_holds"] is True
    assert artifact["efficiency_pareto_hardened"] is True
    assert artifact["verifier_thesis_state"] == "in_generation_moat_holds"
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["cross_generator"]["cross_generator_delta"] == pytest.approx(0.32)
    assert artifact["partial_state_scorer"]["partial_state_auroc"] == pytest.approx(0.81)
    assert artifact["in_generation"]["carnot_minus_rfg_delta"] == pytest.approx(0.21)
    assert artifact["efficiency"]["cost_ratio"] == pytest.approx(0.04)
    assert artifact["self_learning"]["online_adaptation_helps"] is True
    assert artifact["arc_progress"]["total_levels"] == 22
    assert artifact["verifier_is_oracle_honored"] is True

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAMS)
    for key, upstream in mod.DEFAULT_UPSTREAMS.items():
        expected_sha = hashlib.sha256((tmp_path / upstream.path).read_bytes()).hexdigest()
        assert provenance[key]["sha256"] == expected_sha
    assert artifact["reproducibility_checksum"] == mod.checksum_from_provenance(
        artifact["upstream_provenance"]
    )


def test_req_capstone_4301_excludes_flagged_and_blocks_circular_headlines(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4301: flagged and circular results cannot headline moats."""

    payloads = _minimal_payloads()
    payloads["4291_cross_generator"]["verifier_is_oracle"] = True
    payloads["4292_partial_state"]["partial_state_leak_free"] = False
    payloads["4293_generation"]["flagged_adversarial"] = True
    payloads["4294_efficiency"]["verifier_is_oracle"] = True
    _write_default_artifacts(tmp_path, payloads)

    def live_flags(path: Path) -> list[dict[str, str]]:
        if path.name == mod.DEFAULT_UPSTREAMS["4295_self_learning"].path.name:
            return [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}]
        return []

    artifact = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.25,
        live_flag_runner=live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["cross_generator_moat_closes"] is False
    assert artifact["in_generation_moat_holds"] is False
    assert artifact["efficiency_pareto_hardened"] is False
    assert artifact["verifier_thesis_state"] == "partial_state_scorer_leaked"
    assert artifact["verifier_is_oracle_honored"] is False
    assert artifact["oracle_distinct_violations"] == [
        "4291_cross_generator:cross_generator",
        "4294_efficiency:efficiency_pareto",
    ]
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4293_generation", "4295_self_learning"}
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4293_generation"]["fields_imported"] == []
    assert provenance["4295_self_learning"]["fields_imported"] == []


def test_req_capstone_4301_validation_write_and_helpers(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4301: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4301_capstone_v397.json"),
        started_s=4.0,
        now_s=4.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)

    assert mod.bool_metric({"x": True}, "x") is True
    assert mod.bool_metric({"x": 1}, "x") is None
    assert mod.int_metric({"x": 2}, "x") == 2
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": 2}, "x") == pytest.approx(2.0)
    assert mod.float_metric({"x": True}, "x") is None
    assert mod.str_metric({"x": "ok"}, "x") == "ok"
    assert mod.str_metric({"x": 1}, "x") == ""
    assert mod.list_metric({"x": [1]}, "x") == [1]
    assert mod.list_metric({"x": "bad"}, "x") == []
    assert mod.dict_metric({"x": {"y": 1}}, "x") == {"y": 1}
    assert mod.dict_metric({"x": []}, "x") == {}
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "sha256:" + "a" * 64}) == (
        "a" * 64
    )
    assert mod.sha_from_payload_checksum({"reproducibility_checksum": "b" * 64}) == "b" * 64
    assert mod.sha_from_payload_checksum({}) == ""
    assert mod.live_has_critical([{"severity": "critical"}]) is True
    assert mod.live_has_critical([{"severity": "warn"}]) is False
    assert mod.clean_payload({"x": 1}, True) is None
    assert mod.clean_payload({"x": 1}, False) == {"x": 1}
    assert mod.checksum_from_provenance([]) == mod.BLOCKED_CHECKSUM

    bad = json.loads(json.dumps(written))
    bad["cross_generator_moat_closes"] = "true"
    with pytest.raises(ValueError, match="cross_generator_moat_closes"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["skipped"] = True
    with pytest.raises(ValueError, match="skipped upstreams"):
        mod.validate_artifact(bad)


def test_req_capstone_4301_defensive_branches_are_covered(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4301: blocked, skipped, and invalid-shape paths stay explicit."""

    with pytest.raises(ValueError, match="non-object"):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("[]\n", encoding="utf-8")
        mod.read_json_object(bad_json)
    assert mod._metric_from_top_or_pass_rates({"pass_rates": {"vote_at_1": 0.3}}, "vote_at_1")
    assert mod.cross_generator_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.cross_generator_read(None, False)["status"] == "missing_or_excluded"
    assert mod.partial_state_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.partial_state_read(None, False)["status"] == "missing_or_excluded"
    assert mod.generation_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.generation_read(None, False)["status"] == "missing_or_excluded"
    assert mod.efficiency_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.efficiency_read(None, False)["status"] == "missing_or_excluded"
    assert mod.self_learning_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.self_learning_read(None, False)["status"] == "missing_or_excluded"
    assert mod.arc_progress_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.arc_progress_read(None, False)["status"] == "missing_or_excluded"
    assert mod.registry_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.registry_read(None, False)["status"] == "missing_or_excluded"
    assert mod.hardware_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.hardware_read(None, False)["status"] == "missing_or_excluded"

    partial = {"partial_state_scorer_built": True, "partial_state_leak_free": True}
    assert (
        mod.verifier_thesis_state(
            {"cross_generator_moat_closes": True},
            {"partial_state_scorer_built": False},
            {"in_generation_moat_holds": True},
        )
        == "partial_state_blocked"
    )
    assert (
        mod.verifier_thesis_state(
            {"cross_generator_moat_closes": True}, partial, {"in_generation_moat_holds": False}
        )
        == "cross_generator_moat_closed"
    )
    assert (
        mod.verifier_thesis_state(
            {"cross_generator_moat_closes": False}, partial, {"in_generation_moat_holds": False}
        )
        == "cross_generator_open_within_pool_only"
    )
    assert mod._cross_part({"status": "excluded_flagged_adversarial"}) == "excluded_flagged"
    assert mod._partial_part({"status": "excluded_flagged_adversarial"}) == "excluded_flagged"
    assert mod._partial_part({"status": "other"}) == "blocked"
    assert mod._efficiency_part({"status": "excluded_flagged_adversarial"}) == "excluded_flagged"
    assert mod._self_learning_part({"status": "other"}) == "static_ceiling"
    assert mod._oracle_violations(
        {"reported_cross_generator_holds": False},
        {"reported_diffusiongemma_guidance_moat": True, "verifier_is_oracle": True},
        {"reported_efficiency_pareto_holds": False},
    ) == ["4293_generation:in_generation"]

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    (tmp_path / mod.DEFAULT_UPSTREAMS["4294_efficiency"].path).write_text("[]\n", encoding="utf-8")
    blocked = mod.build_artifact(
        tmp_path,
        started_s=7.0,
        now_s=7.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    mod.validate_artifact(blocked)
    assert blocked["missing_upstream_artifacts"] == [
        {
            "artifact_key": "4294_efficiency",
            "experiment_id": 4294,
            "reason": "unparsable_or_non_object",
        }
    ]

    artifact = mod.build_artifact(
        tmp_path,
        started_s=8.0,
        now_s=8.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    artifact = _minimal_payloads()
    fixture_root = tmp_path / "fixture_valid"
    _write_default_artifacts(fixture_root, artifact)
    valid = mod.build_artifact(
        fixture_root,
        started_s=9.0,
        now_s=9.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mutations: list[tuple[str, Any, str]] = [
        ("missing required", lambda d: d.pop("headline_outcome"), "missing required"),
        ("bad verdict", lambda d: d.__setitem__("honest_verdict", "not_terminal"), "terminal"),
        ("bad headline", lambda d: d.__setitem__("headline_outcome", ""), "headline_outcome"),
        ("bad thesis", lambda d: d.__setitem__("verifier_thesis_state", "unknown"), "thesis"),
        ("bad exclusions", lambda d: d.__setitem__("flagged_artifacts_excluded", {}), "excluded"),
        ("bad paper", lambda d: d.__setitem__("paper_ready", None), "paper_ready"),
        ("bad provenance", lambda d: d.__setitem__("upstream_provenance", {}), "provenance"),
        ("bad principles", lambda d: d.__setitem__("field_principles", {}), "principles"),
        (
            "checksum mismatch",
            lambda d: d.__setitem__("reproducibility_checksum", "a" * 64),
            "does not match",
        ),
    ]
    for _, mutate, match in mutations:
        broken = json.loads(json.dumps(valid))
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(broken)

    broken_blocked = mod._blocked_artifact([], [], [], 0.0, 0.0)
    broken_blocked["paper_ready"] = True
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(broken_blocked)

    bad_sha = json.loads(json.dumps(valid))
    bad_sha["upstream_provenance"][0]["sha256"] = "bad"
    bad_sha["reproducibility_checksum"] = mod.checksum_from_provenance(
        bad_sha["upstream_provenance"]
    )
    with pytest.raises(ValueError, match="invalid sha256"):
        mod.validate_artifact(bad_sha)
