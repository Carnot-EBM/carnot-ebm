"""Tests for Exp 4346 .401 verifier scorecard capstone.

Spec refs: REQ-CAPSTONE-4346, SCENARIO-CAPSTONE-4346.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v401_4346 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_support_files(root: Path, total: int = 16) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")
    ops = root / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    (ops / "arc_solve_registry.yaml").write_text(
        f"schema_version: 1\nreproducible_total_levels: {total}\n",
        encoding="utf-8",
    )


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
        "4337_leak_robust_scorer": {
            "honest_verdict": "complete: leak fixture",
            "scorer_leak_audit_passed": True,
            "masked_answer_recovery_auroc": 0.55,
            "process_ranking_auroc": 0.7,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4338_in_generation_moat": {
            "honest_verdict": "complete: replication fixture",
            "in_generation_moat_replicates": True,
            "replication_ci95": [0.1, 0.3],
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "benchmark_n": 240,
            "carnot_minus_best_control_delta": 0.2,
            "carnot_minus_self_reward_smc_delta": 0.15,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4339_e3_ar25": {
            "honest_verdict": "success_e3_ar25_L1_reproduced",
            "game": "ar25",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "plan_executed": True,
            "residual_mismatch_class": "missing_world_model_rule_gap_hidden_undo_stack_action7",
            "verifier_accuracy_per_round": [0.9],
            "verifier_best_accuracy": 0.9,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4340_e3_ka59": {
            "honest_verdict": "complete_e3_ka59_partial_model_0.56",
            "game": "ka59",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "plan_executed": False,
            "residual_mismatch_class": "hidden_step_counter_hud_gap",
            "verifier_accuracy_per_round": [0.5625],
            "verifier_best_accuracy": 0.5625,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4341_e3_sc25": {
            "honest_verdict": "success_e3_sc25_L1_reproduced",
            "game": "sc25",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "plan_executed": True,
            "win_mechanic_cracked": True,
            "verifier_accuracy_per_round": [1.0],
            "verifier_best_accuracy": 1.0,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4342_self_learning": {
            "honest_verdict": "complete: self-learning fixture",
            "learned_encoder_transfer_helps": False,
            "cross_game_state_reduction": 1.006,
            "cross_game_state_reduction_ci95": [1.0, 1.016],
            "n_held_out_levels": 13,
            "n_held_out_games": 11,
            "positive_control_passed": True,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4344_hygiene": {
            "honest_verdict": "complete: hygiene fixture",
            "regression_guard_passed": True,
            "registry_reconciled": True,
            "manifest_reconciled": True,
            "gaps_logged": 3,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4346_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4346: OpenSpec declares the .401 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4346" in spec
    assert "SCENARIO-CAPSTONE-4346" in spec
    assert "experiment_4346_capstone_v401.json" in spec
    assert "blocked_publication_gate_missing" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "publication_gate.py --json" in spec
    assert "verifier_is_oracle:false" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4346_current_artifacts_report_v401_scorecard() -> None:
    """SCENARIO-CAPSTONE-4346: current .401 artifacts settle the gate."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: v401_in_generation_replicated_leak_robust_gate_"
        "MET_oracle_distinct_leak_robust_replicated_arc_levels_17_"
        "e3_reproduced_2_self_learning_open_hygiene_passed"
    )
    assert artifact["in_generation_moat_replicates_headline"] is True
    assert artifact["diffusiongemma_gate_status"] == (
        "MET_oracle_distinct_leak_robust_replicated"
    )
    assert artifact["arc_reproducible_total_levels"] == 17
    assert artifact["verifier_thesis_state"] == "in_generation_moat_replicated_leak_robust"
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["per_axis_gaps"] == []
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["paper_ready"] is True
    assert artifact["in_generation_moat"]["replication_ci95_excludes_zero"] is True
    assert artifact["e3_arc_progress"]["reproduced_levels_total"] == 2
    assert artifact["e3_arc_progress"]["games"]["ar25"]["offline_reproduced"] is True
    assert artifact["e3_arc_progress"]["games"]["ka59"]["reproduced_levels"] == 0
    assert artifact["e3_arc_progress"]["games"]["sc25"]["offline_reproduced"] is True
    assert artifact["self_learning"]["learned_encoder_transfer_helps"] is False
    assert artifact["hygiene"]["regression_guard_passed"] is True

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    expected_sha = hashlib.sha256(
        Path("results/experiment_4338_in_generation_moat_replicate_leak_robust.json").read_bytes()
    ).hexdigest()
    assert provenance["4338_in_generation_moat"]["sha256"] == expected_sha
    assert provenance["4338_in_generation_moat"]["fields_imported"] == list(
        mod.IMPORTED_FIELDS["4338_in_generation_moat"]
    )


def test_req_capstone_4346_missing_axis_does_not_zero_available_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4346: missing artifacts become per-axis gaps only."""

    _write_support_files(tmp_path, total=18)
    payloads = _minimal_payloads()
    payloads.pop("4340_e3_ka59")
    payloads.pop("4344_hygiene")
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
    assert artifact["in_generation_moat_replicates_headline"] is True
    assert artifact["diffusiongemma_gate_status"] == (
        "MET_oracle_distinct_leak_robust_replicated"
    )
    assert artifact["arc_reproducible_total_levels"] == 18
    assert artifact["e3_arc_progress"]["reproduced_levels_total"] == 2
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["per_axis_gaps"] == [
        {"axis": "arc_e3", "artifact_key": "4340_e3_ka59", "experiment_id": 4340},
        {"axis": "hygiene", "artifact_key": "4344_hygiene", "experiment_id": 4344},
    ]
    assert "4338_in_generation_moat" in artifact["availability_report"]["available_artifact_keys"]


def test_req_capstone_4346_flagged_live_critical_and_oracle_are_bounded(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4346: flagged, live-critical, and circular claims are bounded."""

    _write_support_files(tmp_path)
    payloads = _minimal_payloads()
    payloads["4338_in_generation_moat"]["verifier_is_oracle"] = True
    payloads["4342_self_learning"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)

    def live_flags(path: Path) -> list[dict[str, str]]:
        if path.name == mod.DEFAULT_UPSTREAMS["4344_hygiene"].path.name:
            return [{"kind": "DEGENERATE_CONTROLS", "severity": "critical", "detail": "fixture"}]
        return []

    artifact = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        live_flag_runner=live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["in_generation_moat_replicates_headline"] is False
    assert artifact["diffusiongemma_gate_status"] == "STILL_PENDING_verifier_oracle_not_distinct"
    assert artifact["verifier_is_oracle_honored"] is False
    assert artifact["oracle_distinct_violations"] == [
        "4338_in_generation_moat:in_generation_moat"
    ]
    excluded = {row["artifact_key"] for row in artifact["flagged_artifacts_excluded"]}
    assert excluded == {"4342_self_learning", "4344_hygiene"}
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4342_self_learning"]["fields_imported"] == []
    assert provenance["4344_hygiene"]["fields_imported"] == []


def test_req_capstone_4346_publication_gate_missing_blocks_without_aggregation(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4346: missing publication_gate.py stops honestly."""

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_publication_gate_missing"
    assert artifact["diffusiongemma_gate_status"] == "STILL_PENDING_publication_gate_missing"
    assert artifact["verifier_thesis_state"] == "blocked_publication_gate_missing"
    assert artifact["upstream_provenance"] == []
    assert artifact["reproducibility_checksum"] == mod.BLOCKED_PUBLICATION_GATE_CHECKSUM


def test_req_capstone_4346_write_validate_and_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4346: validation, helper edges, and wrapper stay strict."""

    _write_support_files(tmp_path)
    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4346_capstone_v401.json"),
        started_s=5.0,
        now_s=5.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.checksum_from_provenance(
        written["upstream_provenance"]
    )

    assert mod.scorer_leak_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.scorer_leak_read(None, False)["status"] == "missing_or_excluded"
    assert mod.in_generation_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.in_generation_read(None, False)["status"] == "missing_or_excluded"
    assert mod.e3_single_read(None, True, "x")["status"] == "excluded_flagged_adversarial"
    assert mod.e3_single_read(None, False, "x")["status"] == "missing_or_excluded"
    assert mod.self_learning_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.self_learning_read(None, False)["status"] == "missing_or_excluded"
    assert mod.hygiene_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.hygiene_read(None, False)["status"] == "missing_or_excluded"
    assert mod.hygiene_read({"gaps_logged": ["a", "b"]}, False)["gaps_logged_count"] == 2
    assert mod.read_registry_total_levels(tmp_path / "missing") == {
        "status": "missing",
        "reproducible_total_levels": 0,
        "path": "ops/arc_solve_registry.yaml",
    }
    bad_registry = tmp_path / "bad_registry"
    _write_support_files(bad_registry)
    (bad_registry / "ops" / "arc_solve_registry.yaml").write_text("[bad\n", encoding="utf-8")
    assert mod.read_registry_total_levels(bad_registry)["status"] == "unparseable"
    non_mapping_registry = tmp_path / "non_mapping_registry"
    _write_support_files(non_mapping_registry)
    (non_mapping_registry / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")
    assert mod.read_registry_total_levels(non_mapping_registry)["error"] == "non-mapping registry"
    bad_total_registry = tmp_path / "bad_total_registry"
    _write_support_files(bad_total_registry)
    (bad_total_registry / "ops" / "arc_solve_registry.yaml").write_text(
        "reproducible_total_levels: nope\n",
        encoding="utf-8",
    )
    assert "missing or non-int" in mod.read_registry_total_levels(bad_total_registry)["error"]

    assert mod.diffusiongemma_gate_status(
        {"status": "missing_or_excluded"}, {"status": "replicated"}
    ) == "STILL_PENDING_leak_robust_scorer_unavailable"
    assert mod.diffusiongemma_gate_status(
        {"status": "failed", "scorer_leak_audit_passed": False},
        {"status": "replicated"},
    ) == "STILL_PENDING_leak_robust_scorer_failed"
    assert mod.diffusiongemma_gate_status(
        {"status": "passed", "scorer_leak_audit_passed": True, "verifier_is_oracle": True},
        {"status": "replicated"},
    ) == "STILL_PENDING_leak_robust_scorer_oracle_not_distinct"
    assert mod.diffusiongemma_gate_status(
        {"status": "passed", "scorer_leak_audit_passed": True, "verifier_is_oracle": False},
        {"status": "missing_or_excluded"},
    ) == "STILL_PENDING_second_corpus_replication_unavailable"
    assert mod.diffusiongemma_gate_status(
        {"status": "passed", "scorer_leak_audit_passed": True, "verifier_is_oracle": False},
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": True,
            "controls_differentiated": False,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": True,
            "verifier_is_oracle": False,
        },
    ) == "STILL_PENDING_controls_not_differentiated"
    assert mod.diffusiongemma_gate_status(
        {"status": "passed", "scorer_leak_audit_passed": True, "verifier_is_oracle": False},
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": True,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": False,
            "replication_ci95_excludes_zero": True,
            "verifier_is_oracle": False,
        },
    ) == "STILL_PENDING_second_corpus_scorer_leak_recheck_failed"
    assert mod.diffusiongemma_gate_status(
        {"status": "passed", "scorer_leak_audit_passed": True, "verifier_is_oracle": False},
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": True,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": False,
            "verifier_is_oracle": False,
        },
    ) == "STILL_PENDING_ci95_includes_zero"
    assert mod.diffusiongemma_gate_status(
        {"status": "passed", "scorer_leak_audit_passed": True, "verifier_is_oracle": False},
        {
            "status": "retired_corpus_specific",
            "reported_in_generation_moat_replicates": False,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": True,
            "verifier_is_oracle": False,
        },
    ) == "RETIRED_corpus_specific_powered_non_replication"
    assert mod.diffusiongemma_gate_status(
        {"status": "passed", "scorer_leak_audit_passed": True, "verifier_is_oracle": False},
        {
            "status": "measured",
            "reported_in_generation_moat_replicates": True,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "replication_ci95_excludes_zero": True,
            "in_generation_moat_replicates_headline": False,
            "verifier_is_oracle": False,
        },
    ) == "STILL_PENDING_second_corpus_replication_false"

    assert mod.verifier_thesis_state("RETIRED_corpus_specific_x", 2, False) == (
        "in_generation_moat_retired_corpus_specific"
    )
    assert mod.verifier_thesis_state("STILL_PENDING_x", 1, False) == "first_e3_arc_solve"
    assert mod.verifier_thesis_state("STILL_PENDING_x", 0, True) == (
        "verifier_domain_bound_self_learning_open"
    )
    assert mod._oracle_violations(  # noqa: SLF001
        {"scorer_leak_audit_passed": True, "verifier_is_oracle": True},
        {"reported_in_generation_moat_replicates": True, "verifier_is_oracle": True},
        {"reported_learned_encoder_transfer_helps": True, "verifier_is_oracle": True},
    ) == [
        "4337_leak_robust_scorer:scorer_leak_audit",
        "4338_in_generation_moat:in_generation_moat",
        "4342_self_learning:action_role_encoder",
    ]
    assert mod.checksum_from_provenance([]) == mod.EMPTY_UPSTREAM_CHECKSUM
    assert mod._status_part(  # noqa: SLF001
        {"status": "retired_corpus_specific"}, "x", "yes", "no"
    ) == "retired_corpus_specific"

    parse_root = tmp_path / "parse_error"
    _write_support_files(parse_root)
    bad_artifact = parse_root / mod.DEFAULT_UPSTREAMS["4340_e3_ka59"].path
    bad_artifact.parent.mkdir(parents=True, exist_ok=True)
    bad_artifact.write_text("[]\n", encoding="utf-8")
    parse_artifact = mod.build_artifact(
        parse_root,
        started_s=6.0,
        now_s=6.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _paper_gate(True),
    )
    mod.validate_artifact(parse_artifact)
    assert parse_artifact["flagged_artifacts_excluded"][0]["reason"] == (
        "unparsable_or_non_object"
    )

    for field, value, pattern in [
        ("honest_verdict", "not_terminal", "terminal-prefixed"),
        ("in_generation_moat_replicates_headline", "true", "bare bool"),
        ("diffusiongemma_gate_status", "", "diffusiongemma_gate_status"),
        ("arc_reproducible_total_levels", True, "bare int"),
        ("verifier_thesis_state", "unknown", "verifier_thesis_state"),
        ("verifier_is_oracle_honored", None, "verifier_is_oracle_honored"),
        ("per_axis_gaps", {}, "per_axis_gaps"),
        ("flagged_artifacts_excluded", {}, "flagged_artifacts_excluded"),
        ("paper_ready", None, "paper_ready"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
    ]:
        bad = json.loads(json.dumps(written))
        bad[field] = value
        with pytest.raises(ValueError, match=pattern):
            mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="upstream provenance"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["skipped"] = True
    with pytest.raises(ValueError, match="skipped upstreams"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["reproducibility_checksum"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad)

    wrapper = Path("results/experiment_4346_capstone_v401.py").read_text(encoding="utf-8")
    assert "capstone_v401_4346" in wrapper
