"""Tests for REQ-REPORT-4387 / SCENARIO-REPORT-4387."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4387_sota_ingestion_v406 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4387_sota_ingestion_v406.json")
WRAPPER_PATH = Path("results/experiment_4387_sota_ingestion_v406.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v406=mod.DEFAULT_FLAGGED_FOR_V406,
        out_of_band_flagged=mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4387_spec_anchor_exists() -> None:
    """REQ-REPORT-4387: OpenSpec declares the .406 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4387" in spec
    assert "SCENARIO-REPORT-4387" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v406" in spec
    assert "detector_localization_actionable=false" in spec
    assert "detector_compounds=true" in spec
    assert "detector_generalizes_cross_domain=true" in spec
    assert "zero new E3 levels reproduced" in spec


def test_build_artifact_has_required_fields_for_req_report_4387() -> None:
    """REQ-REPORT-4387: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v406": mod.DEFAULT_FLAGGED_FOR_V406,
        "out_of_band_flagged": mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable "
                "citations (or blocked_network_unavailable)."
            ),
            "methods_mapped": (
                "Each method MUST carry a real, VERIFIED arXiv ID/URL (no "
                "citation = fabrication) + a one-line .406 experiment mapping "
                "+ the failure mode + the .405-outcome conditioning."
            ),
            "flagged_for_v406": (
                "Closes discover->ingest->plan: names the single strongest "
                "method for the .406 planner, conditioned on the .405 outcomes."
            ),
            "out_of_band_flagged": (
                "Records A2D2/SEPO (verifier-as-reward generator training) as "
                "operator-owned, NOT auto-run in-loop -- the standing HARD RULE."
            ),
            "random_seed": (
                "Determinism placeholder for the discovery query set "
                "(reproducibility of the sweep)."
            ),
        },
    }


def test_extract_v405_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4387: source artifacts determine the condition."""

    outcomes = mod.extract_v405_outcomes(
        localization={
            "honest_verdict": "complete: clean_powered_null_bidirectional_not_actionable",
            "detector_localization_actionable": False,
            "localization_delta_ci95": [0.0, 0.0],
            "abstention_curve": {"useful_operating_point": None},
            "verifier_is_oracle": False,
        },
        skeptic={"honest_verdict": "blocked_gate_check_failed"},
        compounds={
            "honest_verdict": "success: detector_compounds_heldout_localization_f1",
            "detector_compounds": True,
            "compounding_delta_ci95": [0.003396, 0.032772],
            "verifier_is_oracle": False,
            "fresh_headroom_direction": "cross_domain_detection_exp4386",
        },
        cross_domain={
            "honest_verdict": "success: detector_generalizes_cross_domain_non_fover",
            "detector_generalizes_cross_domain": True,
            "verifier_is_oracle": False,
            "detection_by_domain": [
                {
                    "domain": "gap4_arc",
                    "detection_auroc": 0.963317,
                    "auroc_ci95": [0.922285, 0.990662],
                    "selection_headroom": 0.129,
                    "n": 28443,
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
    )

    assert outcomes == mod.DEFAULT_V405_OUTCOMES


def test_select_flagged_for_v406_conditions_on_v405_outcomes() -> None:
    """SCENARIO-REPORT-4387: .406 flag follows the .405 branch decision."""

    assert (
        mod.select_flagged_for_v406(mod.DEFAULT_V405_OUTCOMES)
        == mod.DEFAULT_FLAGGED_FOR_V406
    )
    assert (
        mod.select_flagged_for_v406(
            mod.DEFAULT_V405_OUTCOMES
            | {
                "biprm_localization_null": False,
                "detector_localization_actionable": True,
                "biprm_abstention_no_useful_point": False,
            }
        )
        == mod.PVD_SELECTIVE_FLAGGED_FOR_V406
    )
    assert (
        mod.select_flagged_for_v406(
            mod.DEFAULT_V405_OUTCOMES
            | {
                "detector_compounds": False,
                "cross_domain_generalizes": True,
                "gap4_arc_ci_lower_gt_chance": True,
            }
        )
        == mod.MULTI_DOMAIN_CALIBRATION_FLAGGED_FOR_V406
    )
    assert (
        mod.select_flagged_for_v406(
            mod.DEFAULT_V405_OUTCOMES
            | {
                "detector_compounds": False,
                "cross_domain_generalizes": False,
                "e3_deeper_new_levels_positive": True,
            }
        )
        == mod.MIND_STUDIO_GAP_REPAIR_FLAGGED_FOR_V406
    )
    assert (
        mod.select_flagged_for_v406(
            mod.DEFAULT_V405_OUTCOMES
            | {
                "detector_compounds": False,
                "cross_domain_generalizes": False,
                "e3_deeper_new_levels_positive": False,
            }
        )
        == mod.BIPRM_BASELINE_ONLY_FLAGGED_FOR_V406
    )


def test_outcome_helpers_reject_malformed_inputs_for_req_report_4387() -> None:
    """REQ-REPORT-4387: malformed source fields never imply a strong branch."""

    assert mod._ci_lower_gt_zero([0.1, 0.2]) is True
    assert mod._ci_lower_gt_zero([0.0, 0.2]) is False
    assert mod._ci_lower_gt_zero("bad") is False
    assert mod._ci_lower_gt_chance([0.51, 0.6]) is True
    assert mod._ci_lower_gt_chance([0.5, 0.6]) is False
    assert mod._ci_lower_gt_chance([]) is False
    assert mod._gap4_arc_domain_generalizes({"detection_by_domain": "bad"}) is False
    assert mod._gap4_arc_domain_generalizes({"detection_by_domain": []}) is False
    assert (
        mod._gap4_arc_domain_generalizes(
            {
                "detection_by_domain": [
                    {
                        "domain": "gap4_arc",
                        "auroc_ci95": [0.51, 0.9],
                        "selection_headroom": 0.01,
                        "n": 1000,
                    }
                ]
            }
        )
        is True
    )
    assert (
        mod._gap4_arc_domain_generalizes(
            {
                "detection_by_domain": [
                    {
                        "domain": "gap4_arc",
                        "auroc_ci95": [0.51, 0.9],
                        "selection_headroom": 0.0,
                        "n": 1000,
                    }
                ]
            }
        )
        is False
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
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "source_verification": "fake",
                        "track": "fake",
                        "v405_outcome_conditioning": "fake",
                        "carnot_stack_mapping": "fake",
                        "failure_mode": "fake",
                        "experiment_mapping": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified source",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"url": "https://example.com/2605.02395"}
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact() | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"experiment_mapping": ""}
                ]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v406": ""}, "flagged_for_v406"),
        (
            _valid_artifact() | {"flagged_for_v406": "unconditioned_followup_v406"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4387"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4387(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4387: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4387: artifact fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["outcomes_mapped"] = {}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {
        "methods_mapped": ["not-a-dict"] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)

    malformed_method_fields = _valid_artifact() | {
        "methods_mapped": [_valid_methods()[0] | {"unexpected": "field"}]
        + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_method_fields)

    malformed_oob = _valid_artifact() | {"out_of_band_flagged": []}
    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_artifact(malformed_oob)


def test_validate_artifact_rejects_out_of_band_row_violations() -> None:
    """REQ-REPORT-4387: A2D2/SEPO out-of-band rows are source-gated."""

    valid_oob = [dict(row) for row in mod.DEFAULT_OUT_OF_BAND_FLAGGED]

    for bad_rows, message in [
        (["bad-row", valid_oob[1]], "exactly"),
        ([valid_oob[0] | {"reason": ""}, valid_oob[1]], "non-empty string"),
        (
            [
                valid_oob[0] | {"arxiv_id_or_url": "9999.99999"},
                valid_oob[1],
            ],
            "not allowed",
        ),
        (
            [
                valid_oob[0] | {"url": "https://example.com/a2d2"},
                valid_oob[1],
            ],
            "url",
        ),
        (
            [
                valid_oob[0] | {"owner_boundary": "auto-run candidate"},
                valid_oob[1],
            ],
            "operator boundary",
        ),
        ([valid_oob[0], valid_oob[0]], "include A2D2 and SEPO"),
    ]:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(
                _valid_artifact() | {"out_of_band_flagged": bad_rows}
            )


def test_validate_studying_section_checks_scenario_report_4387_content() -> None:
    """SCENARIO-REPORT-4387: studying entry maps sources to .406 targets."""

    section = """
    ## 2026-06-18 Exp 4387 - .405 fork SOTA ingestion ingested
    network precondition passed sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch /deep-research not invoked
    complete: clean_powered_null_bidirectional_not_actionable detector_localization_actionable=false localization_delta_ci95=[0.0, 0.0] useful_operating_point=null
    blocked_gate_check_failed
    success: detector_compounds_heldout_localization_f1 detector_compounds=true compounding_delta_ci95=[0.003396, 0.032772]
    success: detector_generalizes_cross_domain_non_fover detector_generalizes_cross_domain=true detection_auroc=0.963317 auroc_ci95=[0.922285, 0.990662] selection_headroom=0.129
    complete_e3_deeper_partial complete_e3_ar25_ka59_ft09_partial new_levels_reproduced=0 reproducible_total_levels=34
    arXiv:2605.02395 arXiv:2102.10395 arXiv:2605.25133 arXiv:2504.16828 arXiv:2606.16070
    out_of_band_flagged arXiv:2606.13565 arXiv:2502.01384 operator-owned NOT auto-run
    flagged_for_v406: verifiable_process_data_cross_domain_localization_v406
    random_seed=4387
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4387: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v406"):
        mod.validate_studying_section("## Fresh pass\narXiv:2605.02395\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2605.02395", "synthetic PRM")
        )

    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("out_of_band_flagged", "operator note")
        )

    with pytest.raises(ValueError, match="out-of-band citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2606.13565", "A2D2")
        )

    with pytest.raises(ValueError, match="detector_compounds=true"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("detector_compounds=true", "detector_null")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4387(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4387: writer emits artifact and studying entry."""

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
    assert studying.count("2026-06-18 Exp 4387") == 1
    assert "flagged_for_v406" in studying
    assert "out_of_band_flagged" in studying
    assert "detector_compounds=true" in studying
    assert "detector_generalizes_cross_domain_non_fover" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4387() -> None:
    """REQ-REPORT-4387: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-18 Exp 4387") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-18 Exp 4387")
    assert studying_refreshed.count("2026-06-18 Exp 4387") == 1
    assert marker_at_end.count("2026-06-18 Exp 4387") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "Controllable and Verifiable Process Data Synthesis" in no_heading


def test_main_prints_terminal_verdict_for_req_report_4387(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4387: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *,
        artifact_path: Path,
        studying_path: Path,
    ) -> dict[str, object]:
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        return {"honest_verdict": mod.DEFAULT_HONEST_VERDICT}

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())


def test_module_main_guard_exits_zero_for_req_report_4387(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4387: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4387_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4387(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4387: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4387_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4387() -> None:
    """REQ-REPORT-4387: committed JSON artifact satisfies the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v406"] == mod.DEFAULT_FLAGGED_FOR_V406
    assert artifact["out_of_band_flagged"] == mod.DEFAULT_OUT_OF_BAND_FLAGGED
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-18 Exp 4387 - .405 fork SOTA ingestion ingested" in studying
    assert (
        "Flagged for .406: `verifiable_process_data_cross_domain_localization_v406`"
    ) in studying
