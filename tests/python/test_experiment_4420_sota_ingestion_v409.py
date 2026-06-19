"""Tests for REQ-REPORT-4420 / SCENARIO-REPORT-4420."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4420_sota_ingestion_v409 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4420_sota_ingestion_v409.json")
WRAPPER_PATH = Path("results/experiment_4420_sota_ingestion_v409.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v409=mod.DEFAULT_FLAGGED_FOR_V409,
        out_of_band_flagged=mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        preconditions_checked=dict(mod.DEFAULT_PRECONDITIONS_CHECKED),
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4420_spec_anchor_exists() -> None:
    """REQ-REPORT-4420: OpenSpec declares the .409 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4420" in spec
    assert "SCENARIO-REPORT-4420" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v409" in spec
    assert "complete_config_rule_partial" in spec
    assert "complete_e3_adaptive_partial" in spec
    assert "hidden_state_localizer_has_nonposition_signal=false" in spec
    assert "sovereign_gap4_gate_holds=true" in spec
    assert "clean_null_steered_confidence_does_not_rescue_code_detector" in spec
    assert "arXiv:2605.05485" in spec


def test_build_artifact_has_required_fields_for_req_report_4420() -> None:
    """REQ-REPORT-4420: artifact exposes the required principle fields."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["flagged_for_v409"] == mod.DEFAULT_FLAGGED_FOR_V409
    assert artifact["methods_mapped"] == _valid_methods()
    assert artifact["out_of_band_flagged"] == mod.DEFAULT_OUT_OF_BAND_FLAGGED
    assert artifact["preconditions_checked"] == mod.DEFAULT_PRECONDITIONS_CHECKED
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES


def test_extract_v408_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4420: source artifacts determine the .409 condition."""

    outcomes = mod.extract_v408_outcomes(
        config_rule={
            "honest_verdict": "complete_config_rule_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
        },
        adaptive_repair={
            "honest_verdict": "complete_e3_adaptive_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
        },
        hidden_state={
            "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
            "hidden_state_localizer_has_nonposition_signal": False,
            "position_only_baseline_f1": 1.0,
            "localization_f1_comparison": {"delta_ci95": [0.0, 0.0]},
            "verifier_is_oracle": False,
        },
        sovereign={
            "honest_verdict": "complete: sovereign_gap4_local_gate_holds_flat_cov_0.2333_fires_0_lost_0",
            "sovereign_gap4_gate_holds": True,
            "pass2_vs_vote": {"graded_gate_fires": 0, "delta_ci95": [0.0, 0.0]},
            "local_generator_coverage": 0.2333,
            "verifier_is_oracle": True,
        },
        vocab_transfer={
            "honest_verdict": "blocked_local_model_unavailable",
            "config_rule_vocabulary_transfers": False,
        },
        steerconf={
            "honest_verdict": "complete: clean_null_steered_confidence_does_not_rescue_code_detector",
            "detection_calibrated_multi_domain": False,
            "domains_at_chance": ["code_humaneval"],
            "positive_control_passed": True,
            "verifier_is_oracle": False,
        },
    )

    assert outcomes == mod.DEFAULT_V408_OUTCOMES


def test_select_flagged_for_v409_conditions_on_v408_outcomes() -> None:
    """SCENARIO-REPORT-4420: .409 flag follows the .408 branch decision."""

    assert mod.select_flagged_for_v409(mod.DEFAULT_V408_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V409
    assert (
        mod.select_flagged_for_v409(
            mod.DEFAULT_V408_OUTCOMES
            | {
                "adaptive_e3_repair_zero_new_levels": False,
                "config_rule_partial_no_new_levels": False,
                "sovereign_gate_flat_no_fires": True,
            }
        )
        == mod.CODEARC_FLAGGED_FOR_V409
    )
    assert (
        mod.select_flagged_for_v409(
            mod.DEFAULT_V408_OUTCOMES
            | {
                "adaptive_e3_repair_zero_new_levels": False,
                "config_rule_partial_no_new_levels": False,
                "sovereign_gate_flat_no_fires": False,
                "code_detector_at_chance_after_steerconf": True,
            }
        )
        == mod.RISCOSET_FLAGGED_FOR_V409
    )
    assert (
        mod.select_flagged_for_v409(
            mod.DEFAULT_V408_OUTCOMES
            | {
                "adaptive_e3_repair_zero_new_levels": True,
                "config_rule_partial_no_new_levels": False,
                "sovereign_gate_flat_no_fires": False,
                "code_detector_at_chance_after_steerconf": False,
            }
        )
        == mod.PREVLA_FLAGGED_FOR_V409
    )
    assert (
        mod.select_flagged_for_v409(
            mod.DEFAULT_V408_OUTCOMES
            | {
                "adaptive_e3_repair_zero_new_levels": False,
                "config_rule_partial_no_new_levels": False,
                "sovereign_gate_flat_no_fires": False,
                "code_detector_at_chance_after_steerconf": False,
            }
        )
        == mod.HIDDEN_AWARENESS_FLAGGED_FOR_V409
    )


def test_outcome_helpers_fail_closed_for_req_report_4420() -> None:
    """REQ-REPORT-4420: malformed optional fields never imply a strong branch."""

    assert mod._ci_equal_zero([0.0, 0.0]) is True
    assert mod._ci_equal_zero([0.0, 0.1]) is False
    assert mod._ci_equal_zero("bad") is False
    assert mod._graded_gate_fires({"pass2_vs_vote": {"graded_gate_fires": 0}}) == 0
    assert mod._graded_gate_fires({"pass2_vs_vote": {"graded_gate_fires": "0"}}) is None
    assert mod._graded_gate_fires({}) is None
    assert mod._contains_string(["code_humaneval"], "code_humaneval") is True
    assert mod._contains_string("code_humaneval", "code_humaneval") is False
    assert mod._contains_string([], "code_humaneval") is False
    assert (
        mod.extract_v408_outcomes(
            config_rule={},
            adaptive_repair={"new_levels_reproduced": 1},
            hidden_state={"localization_f1_comparison": "bad"},
            sovereign={"pass2_vs_vote": "bad"},
            vocab_transfer={},
            steerconf={"domains_at_chance": "bad"},
        )["adaptive_e3_repair_zero_new_levels"]
        is False
    )


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "three to five"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [_valid_methods()[0] | {"arxiv_id_or_url": "9999.99999"}]
                + _valid_methods()[1:]
            },
            "verified source",
        ),
        (
            _valid_artifact() | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0] | {"failure_mode": ""}] + _valid_methods()[1:]},
            "non-empty string",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [_valid_methods()[0] | {"source_verification": "arXiv checked"}]
                + _valid_methods()[1:]
            },
            "source_verification",
        ),
        (_valid_artifact() | {"flagged_for_v409": ""}, "flagged_for_v409"),
        (_valid_artifact() | {"flagged_for_v409": "unconditioned_followup"}, "conditioned"),
        (_valid_artifact() | {"random_seed": "4420"}, "random_seed"),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"trm_training_stood_down": False}},
            "TRM",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4420(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4420: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_oob_and_blocked_shapes() -> None:
    """SCENARIO-REPORT-4420: artifact fields, OOB rows, and blocked path are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact() | {"url": "https://example.com"}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {"methods_mapped": ["not-a-dict"] + _valid_methods()[1:]}
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)

    malformed_method_fields = _valid_artifact() | {
        "methods_mapped": [_valid_methods()[0] | {"url": "https://arxiv.org"}] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_method_fields)

    with pytest.raises(ValueError, match="generator-training"):
        mod.validate_artifact(_valid_artifact() | {"out_of_band_flagged": []})

    valid_oob = [dict(row) for row in mod.DEFAULT_OUT_OF_BAND_FLAGGED]
    for bad_rows, message in [
        (["bad-row"] + valid_oob[1:], "exactly"),
        ([valid_oob[0] | {"reason": ""}] + valid_oob[1:], "non-empty string"),
        ([valid_oob[0] | {"arxiv_id_or_url": "9999.99999"}] + valid_oob[1:], "not allowed"),
        ([valid_oob[0] | {"url": "https://example.com/a2d2"}] + valid_oob[1:], "url"),
        ([valid_oob[0] | {"owner_boundary": "auto-run candidate"}] + valid_oob[1:], "operator boundary"),
        ([valid_oob[0], valid_oob[0], valid_oob[2]], "include"),
    ]:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(_valid_artifact() | {"out_of_band_flagged": bad_rows})

    blocked_artifact = mod.build_blocked_artifact()
    assert blocked_artifact["honest_verdict"] == mod.BLOCKED_HONEST_VERDICT
    assert blocked_artifact["methods_mapped"] == []
    mod.validate_artifact(blocked_artifact)


def test_validate_artifact_rejects_precondition_and_blocked_violations() -> None:
    """REQ-REPORT-4420: unreliable channels and blocked-shape drift fail closed."""

    valid_preconditions = dict(mod.DEFAULT_PRECONDITIONS_CHECKED)

    for preconditions, message in [
        ([], "preconditions_checked"),
        (valid_preconditions | {"deep_research_invoked": True}, "deep-research"),
        (valid_preconditions | {"research_conductor_modified": True}, "research_conductor"),
        (valid_preconditions | {"sweep_clusters_imported": False}, "sweep_clusters"),
        (valid_preconditions | {"sweep_semscholar_ran": False}, "sweep_semscholar"),
        (valid_preconditions | {"websearch_webfetch_reachable": False}, "WebSearch/WebFetch"),
        (valid_preconditions | {"sweep_semscholar_status": ""}, "Semantic Scholar"),
        (valid_preconditions | {"arxiv_api_verified_ids": []}, "verified arXiv ids"),
        (valid_preconditions | {"webfetch_http_200_verified_urls": []}, "HTTP 200 source URLs"),
    ]:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(_valid_artifact() | {"preconditions_checked": preconditions})

    blocked_with_flag = mod.build_blocked_artifact() | {"flagged_for_v409": mod.DEFAULT_FLAGGED_FOR_V409}
    with pytest.raises(ValueError, match="blocked artifact"):
        mod.validate_artifact(blocked_with_flag)


def test_validate_studying_section_checks_scenario_report_4420_content() -> None:
    """SCENARIO-REPORT-4420: studying entry maps sources to .409 targets."""

    mod.validate_studying_section(mod.STUDYING_SECTION)

    with pytest.raises(ValueError, match="flagged_for_v409"):
        mod.validate_studying_section("## Fresh pass\narXiv:2605.05485\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("arXiv:2605.22446", "Pre-VLA"))

    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("out_of_band_flagged", "operator note"))

    with pytest.raises(ValueError, match="out-of-band citations"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("arXiv:2606.13565", "A2D2"))


def test_load_v408_outcomes_reads_repo_relative_sources(tmp_path: Path) -> None:
    """REQ-REPORT-4420: loader reads the six .408 source artifacts."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    source_payloads = {
        "experiment_4414_config_rule_induction_solve.json": {
            "honest_verdict": "complete_config_rule_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
        },
        "experiment_4415_agent2world_adaptive_e3_repair.json": {
            "honest_verdict": "complete_e3_adaptive_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
        },
        "experiment_4416_hidden_state_localizer_falsification_audit.json": {
            "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
            "hidden_state_localizer_has_nonposition_signal": False,
            "position_only_baseline_f1": 1.0,
            "localization_f1_comparison": {"delta_ci95": [0.0, 0.0]},
            "verifier_is_oracle": False,
        },
        "experiment_4417_gap4_local_generator_sovereign_arm.json": {
            "honest_verdict": "complete: sovereign_gap4_local_gate_holds_flat_cov_0.2333_fires_0_lost_0",
            "sovereign_gap4_gate_holds": True,
            "pass2_vs_vote": {"graded_gate_fires": 0, "delta_ci95": [0.0, 0.0]},
            "local_generator_coverage": 0.2333,
            "verifier_is_oracle": True,
        },
        "experiment_4418_config_rule_vocabulary_transfer.json": {
            "honest_verdict": "blocked_local_model_unavailable",
            "config_rule_vocabulary_transfers": False,
        },
        "experiment_4419_steerconf_code_detection_calibration_repair.json": {
            "honest_verdict": "complete: clean_null_steered_confidence_does_not_rescue_code_detector",
            "detection_calibrated_multi_domain": False,
            "domains_at_chance": ["code_humaneval"],
            "positive_control_passed": True,
            "verifier_is_oracle": False,
        },
    }
    for filename, payload in source_payloads.items():
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    assert mod.load_v408_outcomes(tmp_path) == mod.DEFAULT_V408_OUTCOMES


def test_write_outputs_updates_files_idempotently_for_req_report_4420(tmp_path: Path) -> None:
    """REQ-REPORT-4420: writer emits the artifact and studying entry."""

    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    studying = studying_path.read_text(encoding="utf-8")
    assert studying.count("2026-06-19 Exp 4420") == 1
    assert "flagged_for_v409" in studying
    assert "out_of_band_flagged" in studying
    assert "complete_e3_adaptive_partial" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4420() -> None:
    """REQ-REPORT-4420: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-19 Exp 4420") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-19 Exp 4420")
    assert studying_refreshed.count("2026-06-19 Exp 4420") == 1
    assert marker_at_end.count("2026-06-19 Exp 4420") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "ReaComp" in no_heading


def test_main_and_wrappers_emit_terminal_verdict_for_req_report_4420(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4420: module and required wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4420_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")
    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()

    with pytest.raises(SystemExit) as wrapper_exit:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")
    assert wrapper_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()
