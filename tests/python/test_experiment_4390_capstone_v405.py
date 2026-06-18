"""Tests for Exp 4390 .405 milestone capstone.

Spec refs: REQ-CAPSTONE-4390, SCENARIO-CAPSTONE-4390.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import capstone_v405_4390 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_support_files(root: Path, total: int = 34, games: int = 17) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reproducible_total_levels": total,
                "reproducible_total_games": games,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _publication_gate(paper_ready: bool = True) -> JsonDict:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _minimal_payloads(
    *,
    prior_detector: bool = True,
    actionable: bool = False,
    genuine: bool = False,
    compounds: bool = True,
    generalizes: bool = True,
    e3_new_levels: int = 0,
) -> dict[str, JsonDict]:
    return {
        "4379_capstone_v404": {
            "honest_verdict": "complete: v404_detector_positive",
            "detector_beats_chance": prior_detector,
            "detector": {
                "detector_auroc": 0.918304 if prior_detector else 0.5,
                "detector_auroc_ci95": [0.9, 0.93] if prior_detector else [0.45, 0.55],
                "selection_headroom": {"headroom": 0.0},
            },
            "reproducible_total_levels": 34,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4381_localization": {
            "honest_verdict": (
                "success: actionable_localization_and_abstention"
                if actionable
                else "complete: clean_powered_null_bidirectional_not_actionable"
            ),
            "detector_localization_actionable": actionable,
            "localization_f1_by_direction": {
                "bidirectional_fusion": {
                    "f1": 0.42 if actionable else 0.096491,
                    "accuracy": 0.42 if actionable else 0.096491,
                    "exact_match_count": 48 if actionable else 11,
                    "n_error_traces": 114,
                },
                "unidirectional_l2r": {
                    "f1": 0.21 if actionable else 0.096491,
                    "accuracy": 0.21 if actionable else 0.096491,
                    "exact_match_count": 24 if actionable else 11,
                    "n_error_traces": 114,
                },
            },
            "localization_delta_ci95": [0.08, 0.2] if actionable else [0.0, 0.0],
            "abstention_curve": {
                "detector_auroc": 0.98,
                "useful_operating_point": {"coverage": 0.75} if actionable else None,
            },
            "n_traces": 6548,
            "n_error_traces": 114,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4382_skeptic": {
            "honest_verdict": "success: localization_win_is_genuine"
            if genuine
            else "blocked_gate_check_failed",
            "status": "success" if genuine else "blocked",
            "localization_win_is_genuine": genuine,
            "gate_check_summary": "fixture",
            "gates_evaluated": [{"passed": genuine}],
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4383_e3_deeper": {
            "honest_verdict": "complete_e3_deeper_partial",
            "new_levels_reproduced": e3_new_levels,
            "reproducible_total_levels": 34 + e3_new_levels,
            "per_target_scorecard": [
                {
                    "game": "tn36",
                    "new_reproduced_level": 8,
                    "offline_reproduced": e3_new_levels > 0,
                    "prior_best_level": 7,
                    "residual_win_mechanic_gap_class": "fixture_gap",
                }
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4384_e3_blocked": {
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "per_game_scorecard": [
                {
                    "game": "ar25",
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "prior_best_level": 1,
                    "residual_gap_class": "fixture_gap",
                }
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4385_compounds": {
            "honest_verdict": "success: detector_compounds_heldout_localization_f1"
            if compounds
            else "complete: detector_compounding_null",
            "detector_compounds": compounds,
            "learning_curve": [
                {"train_corpus_size": 491, "held_out_localization_f1": 0.371134},
                {"train_corpus_size": 4911, "held_out_localization_f1": 0.387097},
            ],
            "compounding_delta_ci95": [0.003396, 0.032772] if compounds else [-0.01, 0.01],
            "positive_control_passed": True,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4386_generalization": {
            "honest_verdict": "success: detector_generalizes_cross_domain_non_fover"
            if generalizes
            else "complete: detector_domain_bound",
            "detector_generalizes_cross_domain": generalizes,
            "detection_by_domain": [
                {
                    "domain": "gap4_arc",
                    "detection_auroc": 0.963317 if generalizes else 0.5,
                    "auroc_ci95": [0.922285, 0.990662] if generalizes else [0.45, 0.55],
                    "n": 28443,
                }
            ],
            "domains_at_chance": [] if generalizes else ["gap4_arc"],
            "unavailable_domains": [],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4390_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4390: OpenSpec declares the .405 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4390" in spec
    assert "SCENARIO-CAPSTONE-4390" in spec
    assert "experiment_4390_capstone_v405.json" in spec
    assert "blocked_publication_gate_unrunnable" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    assert "publication_gate.py --json" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4390_current_artifacts_report_v405_scorecard() -> None:
    """SCENARIO-CAPSTONE-4390: current .405 artifacts report the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: v405_detector_detects_but_not_actionable_compounds_true_"
        "generalizes_true_arc_levels_34_publication_ready"
    )
    assert artifact["detector_actionable_state"] == "detects_but_not_actionable"
    assert artifact["detector_actionability"]["localization"]["status"] == "not_actionable"
    assert artifact["detector_actionability"]["skeptic_validation"]["status"] == "not_genuine"
    assert artifact["detector_actionability"]["prior_detector"]["detector_beats_chance"] is True
    assert artifact["detector_compounds"] is True
    assert artifact["self_learning"]["compounding_delta_ci95"] == pytest.approx(
        [0.003396, 0.032772]
    )
    assert artifact["detector_generalizes_cross_domain"] is True
    assert artifact["generalization"]["detection_by_domain"][0]["domain"] == "gap4_arc"
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["arc_reproducible_progress"]["new_levels_since_prior"] == 0
    assert artifact["arc_e3_outcomes"]["new_levels_reproduced_from_artifacts"] == 0
    assert artifact["verifier_thesis_state"] == (
        "detector_detects_but_not_actionable_detector_compounds_detection_generalizes"
    )
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["preconditions_checked"]["publication_gate"]["runnable"] is True

    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["4381_localization"]["fields_imported"] == list(
        mod.IMPORTED_FIELDS["4381_localization"]
    )
    assert cited["4385_compounds"]["fields_imported"] == list(
        mod.IMPORTED_FIELDS["4385_compounds"]
    )
    assert artifact["flagged_artifacts_excluded"] == []


def test_req_capstone_4390_missing_axis_does_not_zero_available_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4390: missing upstreams become per-axis gaps only."""

    _write_support_files(tmp_path)
    payloads = _minimal_payloads()
    payloads.pop("4386_generalization")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["detector_actionable_state"] == "detects_but_not_actionable"
    assert artifact["detector_compounds"] is True
    assert artifact["detector_generalizes_cross_domain"] is False
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["per_axis_gaps"] == [
        {
            "axis": "generalization",
            "artifact_key": "4386_generalization",
            "experiment_id": 4386,
        }
    ]


def test_req_capstone_4390_decision_matrix(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4390: headline decisions follow clean upstream gates."""

    _write_support_files(tmp_path, total=35, games=18)
    _write_default_artifacts(
        tmp_path,
        _minimal_payloads(actionable=True, genuine=True, e3_new_levels=1),
    )
    actionable = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(actionable)
    assert actionable["detector_actionable_state"] == "actionable_localization_and_abstention"
    assert actionable["reproducible_total_levels"] == 35
    assert actionable["arc_e3_outcomes"]["games_with_new_reproducible_levels"] == ["tn36"]
    assert (
        actionable["verifier_thesis_state"]
        == "detector_actionable_detector_compounds_detection_generalizes"
    )

    domain_bound_root = tmp_path / "domain_bound"
    _write_support_files(domain_bound_root)
    _write_default_artifacts(
        domain_bound_root,
        _minimal_payloads(prior_detector=False, compounds=False, generalizes=False),
    )
    domain_bound = mod.build_artifact(
        domain_bound_root,
        started_s=4.0,
        now_s=4.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(domain_bound)
    assert domain_bound["detector_actionable_state"] == "open"
    assert domain_bound["detector_compounds"] is False
    assert domain_bound["detector_generalizes_cross_domain"] is False
    assert domain_bound["verifier_thesis_state"] == (
        "detector_actionability_open_detector_compounding_open_detection_domain_bound"
    )


def test_req_capstone_4390_skips_flagged_without_importing_numbers(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4390: flagged artifacts are excluded before metric import."""

    _write_support_files(tmp_path)
    payloads = _minimal_payloads()
    payloads["4385_compounds"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.75,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["detector_actionable_state"] == "detects_but_not_actionable"
    assert artifact["detector_compounds"] is False
    assert artifact["self_learning"]["status"] == "excluded_flagged_adversarial"
    assert "4385_compounds" not in {
        row["artifact_key"] for row in artifact["cited_upstream_artifacts"]
    }
    excluded = {row["artifact_key"]: row for row in artifact["flagged_artifacts_excluded"]}
    assert excluded["4385_compounds"]["reason"] == "flagged_adversarial"
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4385_compounds"]["fields_imported"] == []


def test_req_capstone_4390_publication_gate_unrunnable_blocks(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4390: an unrunnable publication gate stops honestly."""

    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")

    def broken_gate(_: Path) -> JsonDict:
        raise RuntimeError("fixture gate failed")

    artifact = mod.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=broken_gate,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_publication_gate_unrunnable"
    assert artifact["detector_actionable_state"] == "open"
    assert artifact["detector_compounds"] is False
    assert artifact["detector_generalizes_cross_domain"] is False
    assert artifact["reproducible_total_levels"] == 0
    assert artifact["verifier_thesis_state"] == "blocked_publication_gate_unrunnable"
    assert artifact["publication_gate"]["error"].startswith("RuntimeError")
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["publication_gate"]["runnable"] is False


def test_req_capstone_4390_write_validate_and_strict_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4390: wrapper output and validation stay strict."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _minimal_payloads())
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4390_capstone_v405.json"),
        started_s=6.0,
        now_s=6.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    written = json.loads(out_path.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.checksum_from_provenance(
        written["upstream_provenance"]
    )

    assert mod.prior_detector_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.prior_detector_read(None, False)["status"] == "missing_or_excluded"
    assert mod.localization_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.localization_read(None, False)["status"] == "missing_or_excluded"
    assert mod.skeptic_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.skeptic_read(None, False)["status"] == "missing_or_excluded"
    assert mod.self_learning_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.self_learning_read(None, False)["status"] == "missing_or_excluded"
    assert mod.generalization_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.generalization_read(None, False)["status"] == "missing_or_excluded"
    assert mod.arc_progress_read(None, True, "per_game_scorecard")["status"] == (
        "excluded_flagged_adversarial"
    )
    assert mod.arc_progress_read(None, False, "per_game_scorecard")["status"] == (
        "missing_or_excluded"
    )
    arc_with_bad_row = mod.arc_progress_read(
        {
            "new_levels_reproduced": 1,
            "per_game_scorecard": [
                None,
                {"game": "ft09", "offline_reproduced": True, "new_reproduced_level": 2},
            ],
        },
        False,
        "per_game_scorecard",
    )
    assert arc_with_bad_row["games_with_new_reproducible_levels"] == ["ft09"]
    assert mod.read_registry_progress(tmp_path / "missing")["status"] == "missing"
    bad_registry = tmp_path / "bad_registry"
    _write_support_files(bad_registry)
    (bad_registry / "ops" / "arc_solve_registry.yaml").write_text("[bad\n", encoding="utf-8")
    assert mod.read_registry_progress(bad_registry)["status"] == "unparseable"
    non_mapping_registry = tmp_path / "non_mapping_registry"
    _write_support_files(non_mapping_registry)
    (non_mapping_registry / "ops" / "arc_solve_registry.yaml").write_text("[]\n", encoding="utf-8")
    assert mod.read_registry_progress(non_mapping_registry)["error"] == "non-mapping registry"
    bad_total_registry = tmp_path / "bad_total_registry"
    _write_support_files(bad_total_registry)
    (bad_total_registry / "ops" / "arc_solve_registry.yaml").write_text(
        "reproducible_total_levels: nope\nreproducible_total_games: nope\n",
        encoding="utf-8",
    )
    bad_total = mod.read_registry_progress(bad_total_registry)
    assert bad_total["reproducible_total_levels"] == 0
    assert bad_total["reproducible_total_games"] == 0
    assert mod.checksum_from_provenance([]) == mod.EMPTY_UPSTREAM_CHECKSUM
    assert (
        mod._cited_upstream_artifacts(  # noqa: SLF001
            [
                {"skipped": False, "fields_imported": []},
                {"skipped": False, "fields_imported": "not-list"},
            ]
        )
        == []
    )

    missing_gate = mod.build_artifact(
        tmp_path / "no_gate",
        started_s=7.0,
        now_s=7.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(missing_gate)
    assert missing_gate["honest_verdict"] == "blocked_publication_gate_unrunnable"

    non_object_gate_root = tmp_path / "non_object_gate"
    (non_object_gate_root / "scripts").mkdir(parents=True, exist_ok=True)
    (non_object_gate_root / "scripts" / "publication_gate.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    non_object_gate = mod.build_artifact(
        non_object_gate_root,
        started_s=7.5,
        now_s=7.75,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: [],  # type: ignore[return-value]
    )
    mod.validate_artifact(non_object_gate)
    assert non_object_gate["publication_gate"]["error"] == "publication_gate returned non-object"

    parse_root = tmp_path / "parse_error"
    _write_support_files(parse_root)
    bad_artifact = parse_root / mod.DEFAULT_UPSTREAMS["4384_e3_blocked"].path
    bad_artifact.parent.mkdir(parents=True, exist_ok=True)
    bad_artifact.write_text("[]\n", encoding="utf-8")
    parse_artifact = mod.build_artifact(
        parse_root,
        started_s=8.0,
        now_s=8.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(parse_artifact)
    assert parse_artifact["flagged_artifacts_excluded"][0]["reason"] == (
        "unparsable_or_non_object"
    )

    for field, value, pattern in [
        ("honest_verdict", "not_terminal", "terminal-prefixed"),
        ("detector_actionable_state", "maybe", "detector_actionable_state"),
        ("detector_compounds", None, "detector_compounds"),
        ("detector_generalizes_cross_domain", None, "detector_generalizes_cross_domain"),
        ("reproducible_total_levels", True, "bare int"),
        ("verifier_thesis_state", "unknown", "verifier_thesis_state"),
        ("publication_gate", [], "publication_gate"),
        ("verifier_is_oracle", None, "verifier_is_oracle"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
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
    bad["upstream_provenance"] = {}
    with pytest.raises(ValueError, match="upstream_provenance"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"] = ["bad-row"]
    with pytest.raises(ValueError, match="upstream provenance row"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="upstream provenance"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["upstream_provenance"][0]["skipped"] = True
    bad["upstream_provenance"][0]["fields_imported"] = ["detector_compounds"]
    with pytest.raises(ValueError, match="skipped upstreams"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["reproducibility_checksum"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad)

    wrapper = Path("results/experiment_4390_capstone_v405.py").read_text(encoding="utf-8")
    assert "capstone_v405_4390" in wrapper
