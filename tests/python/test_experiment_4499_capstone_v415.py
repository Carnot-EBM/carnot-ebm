"""Tests for Exp 4499 .415 milestone capstone.

Spec refs: REQ-CAPSTONE-4499, SCENARIO-CAPSTONE-4499.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v415_4499 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _fixture_payloads(*, a3_flagged: bool = False) -> dict[str, JsonDict]:
    return {
        "4490_a1_human_replay": {
            "honest_verdict": "complete: blocked_human_replay_corpus_not_cached",
            "heldout_median_actions_before": None,
            "heldout_median_actions_after": None,
            "implied_efficiency_delta": None,
            "solve_rate_dropped": False,
            "trained_on_human_corpus": False,
            "positive_control": {
                "baseline_actions_to_first_levelup": 3,
                "ranked_actions_to_first_levelup": 1,
                "actions_reduced": True,
            },
            "preconditions_checked": {"human_replay_corpus_present": False},
        },
        "4491_a2_trust_energy": {
            "honest_verdict": "success: world_model_trust_energy_beats_first_clears_baseline",
            "baseline_pick_rate": 0.0,
            "trust_energy_pick_rate": 1.0,
            "positive_control_passed": True,
            "hidden_state_games_n": 11,
            "selected_candidates": [{"game": "ar25", "verifier_is_oracle": False}],
            "verifier_is_oracle": False,
            "preconditions_checked": {"offline_arcade_import_smoke": True},
        },
        "4492_a3_energy_loo": {
            "honest_verdict": "success: energy_augmentation_validated_v3_loo_auroc_0.674",
            "baseline_loo_auroc": 0.503,
            "v2_baseline_loo_auroc": 0.503096152732577,
            "v3_loo_auroc": 0.6744657162333668,
            "loo_gate_passed": True,
            "target_loo_auroc": 0.6,
            "feature_class_deltas": {"v3_full": 0.1713695635007898},
            "feature_class_loo_auroc": {"v3_full": 0.6744657162333668},
            "flagged_adversarial": a3_flagged,
            "preconditions_checked": {"seed": 0},
        },
        "4493_a4_hud_register": {
            "honest_verdict": "complete: hud_register_deepen_honest_residual_l2_not_reproduced",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "candidate_reproduction_attempts": [
                {"game": "ar25", "claimed_level": 2, "reached_level": 1, "reproduced": False}
            ],
            "residual_blockers": ["ar25_l2_not_reproduced"],
            "preconditions_checked": {"offline_arcade_import_smoke": True},
        },
        "4494_a5_adapter_l2": {
            "honest_verdict": "complete: cd82_adapter_deepen_l2_honest_residual",
            "target_game": "cd82",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "reproduction_gate": {
                "game": "cd82",
                "claimed_level": 2,
                "reached_level": 1,
                "reproduced": False,
            },
            "residual_blockers": ["cd82_l2_not_reproduced"],
            "preconditions_checked": {"offline_arcade_import_smoke": True},
        },
        "4495_replay_corpus": {
            "honest_verdict": "complete: staged_attributed_mirror_no_weights",
            "training_example_count": 10000,
            "training_shard_count": 3,
            "weights_committed": False,
            "official_license_verified": False,
            "preconditions_checked": {"source_shards_cached": True},
        },
        "4496_scoreboard": {
            "honest_verdict": "complete: submitted_agent_scoreboard_generic_1_of_7_variant_7_of_25",
            "headline_metrics": {
                "submitted_default_heldout_generic_attempted": 7,
                "submitted_default_heldout_generic_solved": 1,
                "submitted_default_heldout_generic_solve_rate": 1 / 7,
                "variant_transfer_attempted": 25,
                "variant_transfer_solved": 7,
                "variant_transfer_rate": 0.28,
            },
            "leaderboard_submission": False,
            "preconditions_checked": {"parity_test_target": "tests/python/test_arc_submitted_agent_parity.py"},
        },
        "4497_hardware": {
            "honest_verdict": "complete: hardware_continuity_audit_4497",
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "preconditions_checked": [{"resource": "kv260_ssh", "available": True}],
        },
        "4498_sota": {
            "honest_verdict": "complete: arc_imitation_sota_415_mapped_for_v416",
            "strongest_for_v416": "flagged_for_v416: DQfD/PER-style human-replay",
            "source_ids": ["1704.03732"],
            "preconditions_checked": {"training_launched": False},
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4499_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4499: OpenSpec declares the .415 capstone contract first."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4499" in spec
    assert "SCENARIO-CAPSTONE-4499" in spec
    assert "results/experiment_4499_capstone_v415.json" in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "reproducible_total_levels" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4499_current_artifacts_answer_headline_questions() -> None:
    """SCENARIO-CAPSTONE-4499: current .415 artifacts produce the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["a1_actions_to_first_levelup_reduction"]["state"] == "heldout_not_measured"
    assert artifact["a1_actions_to_first_levelup_reduction"]["actions_reduction"] is None
    assert artifact["a1_actions_to_first_levelup_reduction"]["efficiency_win"] is False
    assert artifact["a2_trust_energy_oracle_distinct_verdict"]["state"] == (
        "trust_energy_oracle_distinct_pass"
    )
    assert artifact["a2_trust_energy_oracle_distinct_verdict"]["trust_energy_pick_rate"] == 1.0
    assert artifact["a2_trust_energy_oracle_distinct_verdict"]["verifier_is_oracle"] is False
    assert artifact["a3_energy_augmentation_loo_auroc"]["baseline_loo_auroc"] == pytest.approx(
        0.503
    )
    assert artifact["a3_energy_augmentation_loo_auroc"]["v3_loo_auroc"] == pytest.approx(
        0.6744657162333668
    )
    assert artifact["a3_energy_augmentation_loo_auroc"]["beats_0503_baseline"] is True
    assert artifact["a4_a5_l2_banked"] is False
    assert artifact["a4_a5_l2_details"]["any_l2_banked"] is False
    assert artifact["variant_transfer_rate"] == pytest.approx(0.28)
    assert artifact["variant_transfer_scoreboard"]["variant_transfer_solved"] == 7
    assert artifact["verifier_is_oracle"] is False
    assert all(isinstance(claim["verifier_is_oracle"], bool) for claim in artifact["verifier_claims"])
    assert artifact["flagged_artifacts_skipped"] == []
    assert artifact["submitted_to_leaderboard"] is False
    assert "gated_on" not in artifact
    assert "reproducible_total_levels" not in artifact


def test_req_capstone_4499_missing_and_flagged_inputs_are_recorded(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4499: missing/flagged upstreams do not fabricate clean metrics."""

    payloads = _fixture_payloads(a3_flagged=True)
    payloads.pop("4496_scoreboard")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )

    mod.validate_artifact(artifact)
    assert artifact["a3_energy_augmentation_loo_auroc"]["state"] == "excluded_flagged_adversarial"
    assert artifact["a3_energy_augmentation_loo_auroc"]["beats_0503_baseline"] is False
    assert artifact["variant_transfer_rate"] == 0.0
    assert {"axis": "variant_transfer_rate", "artifact_key": "4496_scoreboard", "experiment_id": 4496} in (
        artifact["per_axis_gaps"]
    )
    flagged = next(row for row in artifact["cited_upstream_artifacts"] if row["experiment_id"] == 4492)
    assert flagged["fields_imported"] == []
    assert artifact["flagged_artifacts_skipped"][0]["experiment_id"] == 4492


def test_req_capstone_4499_helper_branches_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4499: helper branches and schema validation fail closed."""

    assert mod.a1_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a1_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a1_read(
        {
            "heldout_median_actions_before": 10,
            "heldout_median_actions_after": 4,
            "solve_rate_dropped": False,
        },
        False,
    )["efficiency_win"] is True
    assert mod.a2_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a2_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a3_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a3_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a3_read({"v3_loo_auroc": 0.5}, False)["feature_class_deltas"] == {}
    assert mod.a3_read({"v3_loo_auroc": 0.5}, False)["feature_class_loo_auroc"] == {}
    assert mod.a4_a5_l2_read(None, None, False, False)["state"] == "missing_or_excluded"
    assert mod.a4_a5_l2_read(None, None, True, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a4_a5_l2_read({"offline_reproduced": True, "reproduced_levels": 1}, None, False, False)[
        "any_l2_banked"
    ] is False
    assert mod.a4_a5_l2_read(
        {"candidate_reproduction_attempts": ["bad-row"]},
        None,
        False,
        False,
    )["hud_register"]["attempts"] == []
    assert mod.variant_transfer_read(None, False)["state"] == "missing_or_excluded"
    assert mod.variant_transfer_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.variant_transfer_read({"headline_metrics": {"variant_transfer_rate": 0.2}}, False)[
        "heldout_generic_solve_rate"
    ] == 0.0
    assert mod.operational_context_read(None, False)["state"] == "missing_or_excluded"
    assert mod.operational_context_read(None, True)["state"] == "excluded_flagged_adversarial"
    both_skipped_claim = mod._verifier_claims(  # noqa: SLF001
        skipped={"4493_a4_hud_register": True, "4494_a5_adapter_l2": True},
        a2={},
        a4_a5={},
    )[-1]
    assert both_skipped_claim["skipped"] is True

    _write_default_artifacts(tmp_path, _fixture_payloads())
    valid = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )
    invalid_cases = [
        ("__delete__honest_verdict", None, "missing required field"),
        ("honest_verdict", "blocked", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("a1_actions_to_first_levelup_reduction", [], "a1_actions_to_first_levelup_reduction"),
        (
            "a2_trust_energy_oracle_distinct_verdict",
            [],
            "a2_trust_energy_oracle_distinct_verdict",
        ),
        ("a3_energy_augmentation_loo_auroc", [], "a3_energy_augmentation_loo_auroc"),
        ("a4_a5_l2_banked", "no", "a4_a5_l2_banked"),
        ("variant_transfer_rate", True, "variant_transfer_rate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("verifier_claims", {}, "verifier_claims"),
        ("flagged_artifacts_skipped", {}, "flagged_artifacts_skipped"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("random_seed", 1, "random_seed"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("field_principles", {}, "field_principles"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("upstream_provenance", [1], "upstream provenance row"),
        ("upstream_provenance", [{"sha256": "bad"}], "invalid sha256"),
        (
            "upstream_provenance",
            [{"sha256": "1" * 64, "skipped": True, "fields_imported": ["x"]}],
            "skipped upstreams",
        ),
        ("verifier_claims", [{"claim": "x"}], "verifier_claims"),
        ("__gated_on__", True, "gated_on"),
        ("__checksum_mismatch__", True, "reproducibility_checksum"),
    ]
    for field, value, message in invalid_cases:
        invalid = json.loads(json.dumps(valid))
        if field == "__delete__honest_verdict":
            invalid.pop("honest_verdict")
        elif field == "__gated_on__":
            invalid["gated_on"] = value
        elif field == "__checksum_mismatch__":
            invalid["reproducibility_checksum"] = "sha256:" + "1" * 64
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)


def test_req_capstone_4499_write_path_records_capstone_recheck(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4499: writer validates and records the live capstone re-check."""

    _write_default_artifacts(tmp_path, _fixture_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=6.0,
        now_s=7.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        capstone_live_flag_runner=lambda _: [
            {"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}
        ],
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["capstone_live_adversarial_recheck"] == {
        "circular_moat_overclaim": True,
        "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
        "status": "critical_flags",
    }


def test_req_capstone_4499_unparseable_input_is_excluded(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4499: unparsable upstreams are excluded before metric import."""

    bad_path = tmp_path / mod.DEFAULT_UPSTREAMS["4490_a1_human_replay"].path
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json", encoding="utf-8")

    raw, provenance, exclusions = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        _clean_live_flags,
        _summarize_zero,
    )

    assert raw["4490_a1_human_replay"] is None
    assert provenance[0]["parse_error"].startswith("JSONDecodeError")
    assert provenance[0]["fields_imported"] == []
    assert exclusions[0]["reason"] == "unparsable_or_non_object"


def test_req_capstone_4499_live_critical_input_is_excluded(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4499: live-critical upstreams import no headline fields."""

    payloads = {"4491_a2_trust_energy": _fixture_payloads()["4491_a2_trust_energy"]}
    _write_default_artifacts(tmp_path, payloads)

    raw, provenance, exclusions = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        lambda _: [{"kind": "TEST_CRITICAL", "severity": "critical"}],
        _summarize_zero,
    )

    assert raw["4491_a2_trust_energy"]["flagged_adversarial"] is True
    assert provenance[0]["live_critical"] is True
    assert provenance[0]["fields_imported"] == []
    assert exclusions[0]["reason"] == "live_critical_adversarial"
    assert exclusions[0]["live_critical_flags"] == [
        {"kind": "TEST_CRITICAL", "severity": "critical"}
    ]
