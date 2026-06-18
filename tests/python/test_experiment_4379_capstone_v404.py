"""Tests for Exp 4379 .404 milestone capstone.

Spec refs: REQ-CAPSTONE-4379, SCENARIO-CAPSTONE-4379.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import capstone_v404_4379 as mod


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
    llm_beats: bool = False,
    contamination_free: bool = False,
    s3_win: bool = False,
    s3_clean_null: bool = False,
    s3_retired: bool = True,
    detector_win: bool = True,
) -> dict[str, JsonDict]:
    return {
        "4370_llm_heuristic": {
            "honest_verdict": (
                "success: llm_heuristic_beats_linear"
                if llm_beats
                else "complete: clean_powered_null_linear_not_beaten"
            ),
            "acceptance_gate_passed": True,
            "llm_heuristic_beats_linear": llm_beats,
            "static_leakage_clean": True,
            "reproduction_gated": True,
            "n_held_out_levels": 9,
            "held_out_actions_by_heuristic": {
                "bfs_baseline": 646,
                "linear": 646,
                "llm_generated": 608 if llm_beats else 646,
            },
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4371_contamination_skeptic": {
            "experiment": 4371,
            "honest_verdict": (
                "success: contamination_free"
                if contamination_free
                else "blocked_gate_check_failed"
            ),
            "win_is_contamination_free": contamination_free,
            "gate_check_summary": "fixture",
            "gates_evaluated": [{"passed": contamination_free}],
        },
        "4372_e3_deeper": {
            "honest_verdict": "success_e3_deeper_lp85_reproduced",
            "new_levels_reproduced": 1,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "lp85",
                    "new_reproduced_level": 5,
                    "offline_reproduced": True,
                    "prior_best_level": 4,
                    "residual_win_mechanic_gap_class": "none",
                },
                {
                    "game": "tn36",
                    "new_reproduced_level": 7,
                    "offline_reproduced": False,
                    "prior_best_level": 7,
                    "residual_win_mechanic_gap_class": "tn36_gap",
                },
            ],
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4373_e3_blocked": {
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 33,
            "verifier_is_oracle": True,
            "per_game_scorecard": [
                {
                    "game": "ar25",
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "prior_best_level": 1,
                    "residual_gap_class": "ar25_gap",
                }
            ],
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4374_diffusiongemma": {
            "experiment": 4374,
            "honest_verdict": (
                "success: useful_generation_gain"
                if s3_win
                else (
                    "complete: clean_null"
                    if s3_clean_null
                    else "retired_in_generation_conversion_unmeasurable"
                )
            ),
            "acceptance_gate": True,
            "s3_guided_beats_control": s3_win,
            "scorer_requalified_leak_clean": s3_win or s3_clean_null,
            "codila_control_differentiates": s3_win or s3_clean_null,
            "s3_gain_ci95": [0.1, 0.2] if s3_win else [-0.02, 0.01],
            "s3_minus_best_of_n_delta": 0.12 if s3_win else 0.0,
            "retirement_gate": {
                "retired": s3_retired,
                "reason": "scorer_leaky_and_codila_not_differentiating",
            },
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4375_detector": {
            "honest_verdict": (
                "complete: detector_beats_chance_zero_selection_headroom_fover"
                if detector_win
                else "complete: detector_null_zero_selection_headroom_fover"
            ),
            "detector_auroc": 0.918304 if detector_win else 0.5,
            "detector_beats_chance": detector_win,
            "detector_auroc_ci95": [0.8, 0.95] if detector_win else [0.45, 0.55],
            "selection_headroom": {"headroom": 0.0, "oracle_at_k": 0.812097, "vote_at_1": 0.812097},
            "n_candidates": 620,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4379_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4379: OpenSpec declares the .404 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4379" in spec
    assert "SCENARIO-CAPSTONE-4379" in spec
    assert "experiment_4379_capstone_v404.json" in spec
    assert "blocked_publication_gate_unrunnable" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    assert "publication_gate.py --json" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4379_current_artifacts_report_v404_scorecard() -> None:
    """SCENARIO-CAPSTONE-4379: current .404 artifacts report the honest scorecard."""

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
        "complete: v404_efficiency_linear_is_settled_arc_levels_34_s3_retired_detector_positive_publication_ready"
    )
    assert artifact["efficiency_moat_state"] == "linear_is_settled"
    assert artifact["efficiency_moat"]["llm_heuristic_beats_linear"] is False
    assert artifact["efficiency_moat"]["status"] == "clean_powered_null"
    assert artifact["s3_moat_utility"] == "retired"
    assert artifact["diffusiongemma"]["status"] == "retired"
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["arc_reproducible_progress"]["new_levels_since_prior"] == 1
    assert artifact["arc_e3_outcomes"]["new_levels_reproduced_from_artifacts"] == 1
    assert artifact["arc_e3_outcomes"]["games_with_new_reproducible_levels"] == ["lp85"]
    assert artifact["detector_beats_chance"] is True
    assert artifact["detector"]["detector_auroc"] == pytest.approx(0.918304)
    assert artifact["detector"]["selection_headroom"]["headroom"] == 0.0
    assert (
        artifact["verifier_thesis_state"]
        == "linear_settled_in_generation_retired_detector_positive"
    )
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["paper_ready"] is True
    assert artifact["unmet_gates"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["preconditions_checked"]["publication_gate"]["runnable"] is True

    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["4370_llm_heuristic"]["fields_imported"] == list(
        mod.IMPORTED_FIELDS["4370_llm_heuristic"]
    )
    assert cited["4374_diffusiongemma"]["fields_imported"] == list(
        mod.IMPORTED_FIELDS["4374_diffusiongemma"]
    )
    assert artifact["flagged_artifacts_excluded"] == []


def test_req_capstone_4379_missing_axis_does_not_zero_available_axes(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4379: missing upstreams become per-axis gaps only."""

    _write_support_files(tmp_path, total=34, games=17)
    payloads = _minimal_payloads()
    payloads.pop("4375_detector")
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
    assert artifact["efficiency_moat_state"] == "linear_is_settled"
    assert artifact["s3_moat_utility"] == "retired"
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["detector_beats_chance"] is False
    assert artifact["paper_ready"] is False
    assert artifact["unmet_gates"] == ["G2"]
    assert artifact["per_axis_gaps"] == [
        {"axis": "detector", "artifact_key": "4375_detector", "experiment_id": 4375}
    ]


def test_req_capstone_4379_decision_matrix(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4379: headline decisions follow clean upstream gates."""

    _write_support_files(tmp_path)
    _write_default_artifacts(
        tmp_path,
        _minimal_payloads(
            llm_beats=True,
            contamination_free=True,
            s3_win=True,
            s3_retired=False,
        ),
    )
    deepened = mod.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(deepened)
    assert deepened["efficiency_moat_state"] == "deepened_stronger_class"
    assert deepened["s3_moat_utility"] == "useful_generation_gain"
    assert (
        deepened["verifier_thesis_state"]
        == "efficiency_moat_deepened_in_generation_converted_detector_positive"
    )

    null_root = tmp_path / "null"
    _write_support_files(null_root)
    _write_default_artifacts(
        null_root,
        _minimal_payloads(s3_clean_null=True, s3_retired=False, detector_win=False),
    )
    null = mod.build_artifact(
        null_root,
        started_s=4.0,
        now_s=4.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mod.validate_artifact(null)
    assert null["efficiency_moat_state"] == "linear_is_settled"
    assert null["s3_moat_utility"] == "proven_but_not_useful"
    assert null["detector_beats_chance"] is False
    assert null["verifier_thesis_state"] == "linear_settled_proven_not_useful"


def test_req_capstone_4379_skips_flagged_without_importing_numbers(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4379: flagged artifacts are excluded before metric import."""

    _write_support_files(tmp_path)
    payloads = _minimal_payloads(llm_beats=True, contamination_free=True)
    payloads["4370_llm_heuristic"]["flagged_adversarial"] = True
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
    assert artifact["efficiency_moat_state"] == "open"
    assert artifact["efficiency_moat"]["status"] == "excluded_flagged_adversarial"
    assert "4370_llm_heuristic" not in {
        row["artifact_key"] for row in artifact["cited_upstream_artifacts"]
    }
    excluded = {row["artifact_key"]: row for row in artifact["flagged_artifacts_excluded"]}
    assert excluded["4370_llm_heuristic"]["reason"] == "flagged_adversarial"
    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert provenance["4370_llm_heuristic"]["fields_imported"] == []


def test_req_capstone_4379_publication_gate_unrunnable_blocks(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4379: an unrunnable publication gate stops honestly."""

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
    assert artifact["efficiency_moat_state"] == "open"
    assert artifact["s3_moat_utility"] == "open"
    assert artifact["reproducible_total_levels"] == 0
    assert artifact["detector_beats_chance"] is False
    assert artifact["verifier_thesis_state"] == "blocked_publication_gate_unrunnable"
    assert artifact["publication_gate"]["error"].startswith("RuntimeError")
    assert artifact["cited_upstream_artifacts"] == []
    assert artifact["preconditions_checked"]["publication_gate"]["runnable"] is False


def test_req_capstone_4379_write_validate_and_strict_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4379: wrapper output and validation stay strict."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _minimal_payloads())
    out_path = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4379_capstone_v404.json"),
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

    assert mod.efficiency_moat_read(None, None, True, False)["status"] == (
        "excluded_flagged_adversarial"
    )
    assert mod.efficiency_moat_read(None, None, False, False)["status"] == "missing_or_excluded"
    assert (
        mod.efficiency_moat_read(
            {
                "llm_heuristic_beats_linear": True,
                "static_leakage_clean": True,
                "reproduction_gated": True,
                "verifier_is_oracle": False,
            },
            {"win_is_contamination_free": False},
            False,
            False,
        )["status"]
        == "open"
    )
    assert mod.diffusiongemma_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.diffusiongemma_read(None, False)["status"] == "missing_or_excluded"
    assert mod.diffusiongemma_read({}, False)["status"] == "open"
    assert mod.detector_read(None, True)["status"] == "excluded_flagged_adversarial"
    assert mod.detector_read(None, False)["status"] == "missing_or_excluded"
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
    bad_artifact = parse_root / mod.DEFAULT_UPSTREAMS["4373_e3_blocked"].path
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
        ("efficiency_moat_state", "maybe", "efficiency_moat_state"),
        ("s3_moat_utility", "maybe", "s3_moat_utility"),
        ("reproducible_total_levels", True, "bare int"),
        ("detector_beats_chance", None, "bare bool"),
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
    bad["upstream_provenance"][0]["fields_imported"] = ["llm_heuristic_beats_linear"]
    with pytest.raises(ValueError, match="skipped upstreams"):
        mod.validate_artifact(bad)

    bad = json.loads(json.dumps(written))
    bad["reproducibility_checksum"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        mod.validate_artifact(bad)

    wrapper = Path("results/experiment_4379_capstone_v404.py").read_text(encoding="utf-8")
    assert "capstone_v404_4379" in wrapper
