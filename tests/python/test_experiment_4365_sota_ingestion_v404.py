"""Tests for REQ-REPORT-4365 / SCENARIO-REPORT-4365."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4365_sota_ingestion_v404 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4365_sota_ingestion_v404.json")
WRAPPER_PATH = Path("results/experiment_4365_sota_ingestion_v404.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v404=mod.DEFAULT_FLAGGED_FOR_V404,
        out_of_band_flagged=mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4365_spec_anchor_exists() -> None:
    """REQ-REPORT-4365: OpenSpec declares the .404 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4365" in spec
    assert "SCENARIO-REPORT-4365" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v404" in spec
    assert "honest_verdict=scorer_leaky_in_search_corpus" in spec
    assert "reproducible_total_levels>=33" in spec
    assert "action_efficiency_compounds=true" in spec
    assert "verifier_is_oracle=false" in spec


def test_build_artifact_has_required_fields_for_req_report_4365() -> None:
    """REQ-REPORT-4365: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v404": mod.DEFAULT_FLAGGED_FOR_V404,
        "out_of_band_flagged": mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable "
                "citations (or blocked_network_unavailable)."
            ),
            "methods_mapped": (
                "Each method MUST carry a real, VERIFIED arXiv ID/URL (no "
                "citation = fabrication) + a one-line .404 experiment mapping "
                "+ the failure mode + the .403-outcome conditioning."
            ),
            "flagged_for_v404": (
                "Closes discover->ingest->plan: names the single strongest "
                "method for the .404 planner, conditioned on the .403 outcomes."
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


def test_select_flagged_for_v404_conditions_on_v403_outcomes() -> None:
    """SCENARIO-REPORT-4365: .404 flag follows the .403 branch decision."""

    assert mod.select_flagged_for_v404(mod.DEFAULT_V403_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V404
    assert (
        mod.select_flagged_for_v404(
            mod.DEFAULT_V403_OUTCOMES
            | {
                "action_efficiency_compounds": False,
                "action_deployed_into_solver_kit": False,
                "e3_new_levels_positive": True,
            }
        )
        == mod.E3_DEEPER_FLAGGED_FOR_V404
    )
    assert (
        mod.select_flagged_for_v404(
            mod.DEFAULT_V403_OUTCOMES
            | {
                "action_efficiency_compounds": False,
                "e3_new_levels_positive": False,
                "search_acceptance_gate": True,
                "search_scorer_leaky": False,
                "search_controls_differentiated": True,
                "search_guided_beats_control": True,
            }
        )
        == mod.PRISM_REPAIR_FLAGGED_FOR_V404
    )
    assert (
        mod.select_flagged_for_v404(
            mod.DEFAULT_V403_OUTCOMES
            | {
                "action_efficiency_compounds": False,
                "e3_new_levels_positive": False,
                "search_acceptance_gate": True,
                "search_scorer_leaky": True,
            }
        )
        == mod.CODILA_SCORER_QUARANTINE_FLAGGED_FOR_V404
    )
    assert (
        mod.select_flagged_for_v404(
            mod.DEFAULT_V403_OUTCOMES
            | {
                "action_efficiency_compounds": False,
                "e3_new_levels_positive": False,
                "search_acceptance_gate": False,
                "search_scorer_leaky": False,
            }
        )
        == mod.PAPO_DIAGNOSTIC_FLAGGED_FOR_V404
    )


def test_extract_v403_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4365: source artifacts determine the condition."""

    outcomes = mod.extract_v403_outcomes(
        search={
            "honest_verdict": "scorer_leaky_in_search_corpus",
            "acceptance_gate": True,
            "benchmark_n": 0,
            "controls_differentiated": False,
            "s3_guided_beats_control": False,
            "independent_leak_recheck": {
                "scorer_leak_recheck_passed": False,
                "fresh_heldout_n": 240,
            },
        },
        e3={
            "honest_verdict": "success_e3_deeper_tu93_reproduced",
            "new_levels_reproduced": 1,
            "reproducible_total_levels": 33,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "tu93",
                    "offline_reproduced": True,
                    "checkpoint_status": "new_level_reproduced",
                }
            ],
        },
        action={
            "honest_verdict": "success: action_efficiency_compounds_25_to_16",
            "acceptance_gate_passed": True,
            "action_efficiency_compounds": True,
            "deployed_into_solver_kit": True,
            "positive_control_passed": True,
            "reproduction_gated": True,
            "verifier_is_oracle": False,
            "llm_heuristic_arm": {"ran": False, "static_analysis_clean": True},
            "compounding_curve": [
                {"corpus_size_k": 4, "held_out_actions_to_solve": 25},
                {"corpus_size_k": 19, "held_out_actions_to_solve": 16},
            ],
        },
    )

    assert outcomes == mod.DEFAULT_V403_OUTCOMES


def test_outcome_helpers_reject_malformed_inputs_for_req_report_4365() -> None:
    """REQ-REPORT-4365: malformed source fields never imply a strong branch."""

    assert mod._scorer_leak_failed({"scorer_leak_recheck_passed": False}) is True
    assert mod._scorer_leak_failed({"scorer_leak_recheck_passed": True}) is False
    assert mod._scorer_leak_failed("not-a-dict") is False
    assert mod._tu93_new_level_reproduced({"per_target_scorecard": 1}) is False
    assert mod._tu93_new_level_reproduced({"per_target_scorecard": "bad"}) is False
    assert mod._tu93_new_level_reproduced({"per_target_scorecard": []}) is False
    assert mod._curve_reduces_actions("not-a-list") is False
    assert mod._curve_reduces_actions([{"held_out_actions_to_solve": 16}]) is False


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
                        "v403_outcome_conditioning": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2503.18809"}
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
        (_valid_artifact() | {"flagged_for_v404": ""}, "flagged_for_v404"),
        (
            _valid_artifact() | {"flagged_for_v404": "unconditioned_followup_v404"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4365"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4365(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4365: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4365: artifact fields are exact."""

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
        "methods_mapped": [
            _valid_methods()[0] | {"unexpected": "field"}
        ]
        + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_method_fields)

    malformed_oob = _valid_artifact() | {"out_of_band_flagged": []}
    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_artifact(malformed_oob)


def test_validate_artifact_rejects_out_of_band_row_violations() -> None:
    """REQ-REPORT-4365: A2D2/SEPO out-of-band rows are source-gated."""

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


def test_validate_studying_section_checks_scenario_report_4365_content() -> None:
    """SCENARIO-REPORT-4365: studying entry maps sources to .404 targets."""

    section = """
    ## 2026-06-18 Exp 4365 - .403 fork SOTA ingestion ingested
    network precondition passed sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch /deep-research not invoked
    honest_verdict=scorer_leaky_in_search_corpus acceptance_gate=true benchmark_n=0 controls_differentiated=false s3_guided_beats_control=false
    success_e3_deeper_tu93_reproduced new_levels_reproduced=1 reproducible_total_levels=33 verifier_is_oracle=true
    action_efficiency_compounds=true acceptance_gate_passed=true deployed_into_solver_kit=true reproduction_gated=true verifier_is_oracle=false
    arXiv:2503.18809 arXiv:2605.05138 arXiv:2603.20216 arXiv:2606.08501 arXiv:2602.01842
    out_of_band_flagged arXiv:2606.13565 arXiv:2502.01384 operator-owned NOT auto-run
    flagged_for_v404: llm_generated_action_heuristics_compounding_v404
    random_seed=4365
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4365: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v404"):
        mod.validate_studying_section("## Fresh pass\narXiv:2503.18809\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2503.18809", "LLM heuristics")
        )

    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("out_of_band_flagged", "operator note")
        )

    with pytest.raises(ValueError, match="out-of-band citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2606.13565", "A2D2")
        )

    with pytest.raises(ValueError, match="action_efficiency_compounds=true"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "action_efficiency_compounds=true",
                "action_efficiency_compounds=false",
            )
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4365(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4365: writer emits artifact and studying entry."""

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
    assert studying.count("2026-06-18 Exp 4365") == 1
    assert "flagged_for_v404" in studying
    assert "out_of_band_flagged" in studying
    assert "scorer_leaky_in_search_corpus" in studying
    assert "action_efficiency_compounds=true" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4365() -> None:
    """REQ-REPORT-4365: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-18 Exp 4365") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-18 Exp 4365")
    assert studying_refreshed.count("2026-06-18 Exp 4365") == 1
    assert marker_at_end.count("2026-06-18 Exp 4365") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "LLM-Generated Heuristics" in no_heading


def test_main_prints_terminal_verdict_for_req_report_4365(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4365: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4365(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4365: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4365_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4365(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4365: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4365_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4365() -> None:
    """REQ-REPORT-4365: committed JSON artifact satisfies the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v404"] == mod.DEFAULT_FLAGGED_FOR_V404
    assert artifact["out_of_band_flagged"] == mod.DEFAULT_OUT_OF_BAND_FLAGGED
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-18 Exp 4365 - .403 fork SOTA ingestion ingested" in studying
    assert (
        "Flagged for .404: `llm_generated_action_heuristics_compounding_v404`"
    ) in studying
