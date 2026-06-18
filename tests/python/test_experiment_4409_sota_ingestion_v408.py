"""Tests for REQ-REPORT-4409 / SCENARIO-REPORT-4409."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4409_sota_ingestion_v408 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4409_sota_ingestion_v408.json")
WRAPPER_PATH = Path("results/experiment_4409_sota_ingestion_v408.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v408=mod.DEFAULT_FLAGGED_FOR_V408,
        out_of_band_flagged=mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4409_spec_anchor_exists() -> None:
    """REQ-REPORT-4409: OpenSpec declares the .408 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4409" in spec
    assert "SCENARIO-REPORT-4409" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v408" in spec
    assert "clean_powered_null_position_only_not_beaten" in spec
    assert "blocked_gate_check_failed" in spec
    assert "clean_null_position_bound_or_saturated" in spec
    assert "calibrated_multi_domain_contract_false_deconfounded" in spec
    assert "zero new E3 levels reproduced" in spec


def test_build_artifact_has_required_fields_for_req_report_4409() -> None:
    """REQ-REPORT-4409: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["flagged_for_v408"] == mod.DEFAULT_FLAGGED_FOR_V408
    assert artifact["methods_mapped"] == _valid_methods()
    assert artifact["out_of_band_flagged"] == mod.DEFAULT_OUT_OF_BAND_FLAGGED
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES


def test_extract_v407_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4409: source artifacts determine the condition."""

    outcomes = mod.extract_v407_outcomes(
        real_localizer={
            "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
            "localizer_genuinely_beats_position_only": False,
            "verifier_is_oracle": False,
            "localization_f1_by_domain": {
                "FoVer": {
                    "position_only_baseline": 1.0,
                    "real_intervention_localizer": 1.0,
                    "beats_position_only_baseline": False,
                    "delta_ci95": [0.0, 0.0],
                },
                "GAP-4 ARC": {
                    "position_only_baseline": 0.788462,
                    "real_intervention_localizer": 0.807692,
                    "beats_position_only_baseline": False,
                    "delta_ci95": [-0.134615, 0.173077],
                },
            },
        },
        typed_taxonomy={
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "artifact_field": "localizer_genuinely_beats_position_only",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                }
            ],
        },
        e3_deeper={
            "honest_verdict": "complete_e3_deeper_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
        },
        e3_tails={
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "verifier_is_oracle": True,
        },
        active_learning={
            "honest_verdict": "complete: clean_null_position_bound_or_saturated",
            "localizer_compounds": False,
            "compounding_delta_ci95": [0.0, 0.0],
            "positive_control_passed": False,
            "gate_summary": {
                "active_rises_beyond_random": False,
                "positive_control_headroom": False,
            },
            "verifier_is_oracle": False,
        },
        calibration={
            "honest_verdict": "complete: calibrated_multi_domain_contract_false_deconfounded",
            "detection_calibrated_multi_domain": False,
            "domains_at_chance": ["code_humaneval"],
            "positive_control_passed": True,
            "verifier_is_oracle": False,
        },
    )

    assert outcomes == mod.DEFAULT_V407_OUTCOMES


def test_select_flagged_for_v408_conditions_on_v407_outcomes() -> None:
    """SCENARIO-REPORT-4409: .408 flag follows the .407 branch decision."""

    assert mod.select_flagged_for_v408(mod.DEFAULT_V407_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V408
    assert (
        mod.select_flagged_for_v408(
            mod.DEFAULT_V407_OUTCOMES
            | {
                "e3_static_unit_tests_stalled": False,
                "localizer_position_only_null": True,
            }
        )
        == mod.GEOREASON_FLAGGED_FOR_V408
    )
    assert (
        mod.select_flagged_for_v408(
            mod.DEFAULT_V407_OUTCOMES
            | {
                "e3_static_unit_tests_stalled": False,
                "localizer_position_only_null": False,
                "calibration_contract_false": True,
            }
        )
        == mod.STEERCONF_FLAGGED_FOR_V408
    )
    assert (
        mod.select_flagged_for_v408(
            mod.DEFAULT_V407_OUTCOMES
            | {
                "e3_static_unit_tests_stalled": False,
                "localizer_position_only_null": False,
                "calibration_contract_false": False,
                "active_selection_compounds": False,
            }
        )
        == mod.CAPO_DIAGNOSTIC_FLAGGED_FOR_V408
    )


def test_outcome_helpers_reject_malformed_inputs_for_req_report_4409() -> None:
    """REQ-REPORT-4409: malformed source fields never imply a strong branch."""

    assert mod._ci_equal_zero([0.0, 0.0]) is True
    assert mod._ci_equal_zero([0.0, 0.1]) is False
    assert mod._ci_crosses_zero([-0.1, 0.2]) is True
    assert mod._ci_crosses_zero([0.1, 0.2]) is False
    assert mod._ci_crosses_zero("bad") is False
    assert mod._domain_ties_position_baseline({}, "FoVer") is False
    assert (
        mod._domain_ties_position_baseline({"localization_f1_by_domain": {"FoVer": "bad"}}, "FoVer")
        is False
    )
    assert (
        mod._domain_ties_position_baseline(
            {
                "localization_f1_by_domain": {
                    "FoVer": {
                        "position_only_baseline": 1.0,
                        "real_intervention_localizer": 1.0,
                        "beats_position_only_baseline": False,
                        "delta_ci95": [0.0, 0.0],
                    }
                }
            },
            "FoVer",
        )
        is True
    )
    assert mod._domain_delta_crosses_zero({}, "GAP-4 ARC") is False
    assert (
        mod._domain_delta_crosses_zero(
            {"localization_f1_by_domain": {"GAP-4 ARC": "bad"}},
            "GAP-4 ARC",
        )
        is False
    )
    assert mod._gate_failed_for_field({}, "localizer_genuinely_beats_position_only") is False
    assert (
        mod._gate_failed_for_field(
            {
                "gates_evaluated": [
                    {
                        "artifact_field": "localizer_genuinely_beats_position_only",
                        "passed": False,
                    }
                ]
            },
            "localizer_genuinely_beats_position_only",
        )
        is True
    )
    assert (
        mod._gate_failed_for_field(
            {"gates_evaluated": [{"artifact_field": "other", "passed": False}]},
            "localizer_genuinely_beats_position_only",
        )
        is False
    )


def test_extract_v407_outcomes_tolerates_malformed_optional_fields() -> None:
    """REQ-REPORT-4409: optional nested outcome fields fail closed."""

    outcomes = mod.extract_v407_outcomes(
        real_localizer={},
        typed_taxonomy={},
        e3_deeper={"new_levels_reproduced": 1, "reproducible_total_levels": 35},
        e3_tails={"new_levels_reproduced": 1, "reproducible_total_levels": 35},
        active_learning={"gate_summary": "bad"},
        calibration={"domains_at_chance": "bad"},
    )

    assert outcomes["active_positive_control_headroom"] is False
    assert outcomes["code_humaneval_at_chance"] is False
    assert outcomes["e3_deeper_new_levels_positive"] is True
    assert outcomes["e3_tails_new_levels_positive"] is True
    assert outcomes["e3_static_unit_tests_stalled"] is False
    assert (
        mod.select_flagged_for_v408(
            {
                "e3_static_unit_tests_stalled": False,
                "localizer_position_only_null": False,
                "calibration_contract_false": False,
                "active_selection_compounds": True,
            }
        )
        == mod.AERA_FLAGGED_FOR_V408
    )


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (
            _valid_artifact() | {"field_principles": {"honest_verdict": "loose"}},
            "field_principles",
        ),
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
            | {
                "methods_mapped": [_valid_methods()[0] | {"experiment_mapping": ""}]
                + _valid_methods()[1:]
            },
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
        (_valid_artifact() | {"flagged_for_v408": ""}, "flagged_for_v408"),
        (
            _valid_artifact() | {"flagged_for_v408": "unconditioned_followup_v408"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4409"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4409(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4409: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4409: artifact fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["url"] = "https://example.com"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {
        "methods_mapped": ["not-a-dict"] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)

    malformed_method_fields = _valid_artifact() | {
        "methods_mapped": [_valid_methods()[0] | {"url": "https://arxiv.org"}]
        + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_method_fields)

    malformed_oob = _valid_artifact() | {"out_of_band_flagged": []}
    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_artifact(malformed_oob)

    blocked_artifact = mod.build_blocked_artifact()
    assert blocked_artifact["honest_verdict"] == mod.BLOCKED_HONEST_VERDICT
    assert blocked_artifact["methods_mapped"] == []
    mod.validate_artifact(blocked_artifact)


def test_validate_artifact_rejects_out_of_band_row_violations() -> None:
    """REQ-REPORT-4409: A2D2/SEPO out-of-band rows are source-gated."""

    valid_oob = [dict(row) for row in mod.DEFAULT_OUT_OF_BAND_FLAGGED]

    for bad_rows, message in [
        (["bad-row", valid_oob[1]], "exactly"),
        ([valid_oob[0] | {"reason": ""}, valid_oob[1]], "non-empty string"),
        ([valid_oob[0] | {"arxiv_id_or_url": "9999.99999"}, valid_oob[1]], "not allowed"),
        ([valid_oob[0] | {"url": "https://example.com/a2d2"}, valid_oob[1]], "url"),
        (
            [valid_oob[0] | {"owner_boundary": "auto-run candidate"}, valid_oob[1]],
            "operator boundary",
        ),
        ([valid_oob[0], valid_oob[0]], "include A2D2 and SEPO"),
    ]:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(_valid_artifact() | {"out_of_band_flagged": bad_rows})


def test_validate_studying_section_checks_scenario_report_4409_content() -> None:
    """SCENARIO-REPORT-4409: studying entry maps sources to .408 targets."""

    section = """
    ## 2026-06-18 Exp 4409 - .407 fork SOTA ingestion ingested
    reliable channel reachable sweep_clusters.py sweep_semscholar.py HTTP 429 WebSearch/WebFetch /deep-research not invoked
    complete: clean_powered_null_position_only_not_beaten localizer_genuinely_beats_position_only=false position_only_baseline=1.0
    blocked_gate_check_failed localizer_genuinely_beats_position_only actual=False expected=True
    complete: clean_null_position_bound_or_saturated localizer_compounds=false compounding_delta_ci95=[0.0, 0.0] positive_control_headroom=false
    complete: calibrated_multi_domain_contract_false_deconfounded detection_calibrated_multi_domain=false domains_at_chance=[code_humaneval] positive_control_passed=true
    complete_e3_deeper_partial complete_e3_ar25_ka59_ft09_partial new_levels_reproduced=0 reproducible_total_levels=34
    arXiv:2512.22336 arXiv:2605.13772 arXiv:2503.02863 arXiv:2605.25931 arXiv:2508.02298
    out_of_band_flagged arXiv:2606.13565 arXiv:2502.01384 operator-owned NOT auto-run
    flagged_for_v408: agent2world_adaptive_e3_mechanic_repair_v408
    random_seed=4409
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4409: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v408"):
        mod.validate_studying_section("## Fresh pass\narXiv:2512.22336\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2512.22336", "Agent2World")
        )

    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("out_of_band_flagged", "operator note")
        )

    with pytest.raises(ValueError, match="out-of-band citations"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("arXiv:2606.13565", "A2D2"))

    with pytest.raises(ValueError, match="localizer_genuinely_beats_position_only=false"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "localizer_genuinely_beats_position_only=false",
                "localizer_genuinely_beats_position_only=true",
            )
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4409(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4409: writer emits artifact and studying entry."""

    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(
        artifact_path=artifact_path,
        studying_path=studying_path,
    )

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    saved_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    studying = studying_path.read_text(encoding="utf-8")

    assert saved_artifact == artifact
    assert studying.count("2026-06-18 Exp 4409") == 1
    assert "flagged_for_v408" in studying
    assert "out_of_band_flagged" in studying
    assert "localizer_genuinely_beats_position_only=false" in studying
    assert "detection_calibrated_multi_domain=false" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4409() -> None:
    """REQ-REPORT-4409: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-18 Exp 4409") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-18 Exp 4409")
    assert studying_refreshed.count("2026-06-18 Exp 4409") == 1
    assert marker_at_end.count("2026-06-18 Exp 4409") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "Agent2World" in no_heading


def test_main_prints_terminal_verdict_for_req_report_4409(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4409: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *,
        artifact_path: Path,
        studying_path: Path,
        outcomes: object,
    ) -> dict[str, object]:
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        assert outcomes == mod.DEFAULT_V407_OUTCOMES
        return {"honest_verdict": mod.DEFAULT_HONEST_VERDICT}

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())


def test_module_main_guard_exits_zero_for_req_report_4409(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4409: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4409_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4409(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4409: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4409_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()
