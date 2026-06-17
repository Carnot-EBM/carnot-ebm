"""Tests for REQ-REPORT-4354 / SCENARIO-REPORT-4354."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4354_sota_ingestion_v403 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4354_sota_ingestion_v403.json")
WRAPPER_PATH = Path("results/experiment_4354_sota_ingestion_v403.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v403=mod.DEFAULT_FLAGGED_FOR_V403,
        out_of_band_flagged=mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4354_spec_anchor_exists() -> None:
    """REQ-REPORT-4354: OpenSpec declares the .403 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4354" in spec
    assert "SCENARIO-REPORT-4354" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v403" in spec
    assert "out_of_band_flagged" in spec
    assert "acceptance_gate=true" in spec
    assert "new_levels_reproduced=1" in spec
    assert "held_out_actions_baseline=25" in spec


def test_build_artifact_has_required_fields_for_req_report_4354() -> None:
    """REQ-REPORT-4354: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v403": mod.DEFAULT_FLAGGED_FOR_V403,
        "out_of_band_flagged": mod.DEFAULT_OUT_OF_BAND_FLAGGED,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable "
                "citations (or blocked_network_unavailable)."
            ),
            "methods_mapped": (
                "Each method MUST carry a real, VERIFIED arXiv ID/URL (no "
                "citation = fabrication) + a one-line .403 experiment mapping "
                "+ the failure mode + the .402-outcome conditioning."
            ),
            "flagged_for_v403": (
                "Closes discover->ingest->plan: names the single strongest "
                "method for the .403 planner, conditioned on the .402 outcomes."
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


def test_select_flagged_for_v403_conditions_on_v402_outcomes() -> None:
    """SCENARIO-REPORT-4354: .403 flag follows the .402 branch decision."""

    assert mod.select_flagged_for_v403(mod.DEFAULT_V402_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V403
    assert (
        mod.select_flagged_for_v403(
            mod.DEFAULT_V402_OUTCOMES
            | {
                "s3_acceptance_gate": True,
                "s3_adversarial_tautology_flagged": False,
            }
        )
        == mod.S3_DIVERSITY_AUDIT_FLAGGED_FOR_V403
    )
    assert (
        mod.select_flagged_for_v403(
            mod.DEFAULT_V402_OUTCOMES
            | {
                "s3_acceptance_gate": False,
                "action_efficiency_improves": True,
                "action_reduction_reproduced": True,
                "action_verifier_non_oracle": True,
            }
        )
        == mod.ACTION_HEURISTIC_FLAGGED_FOR_V403
    )
    assert (
        mod.select_flagged_for_v403(
            mod.DEFAULT_V402_OUTCOMES
            | {
                "s3_acceptance_gate": False,
                "action_efficiency_improves": False,
                "e3_new_levels_positive": True,
            }
        )
        == mod.E3_DEEPER_FLAGGED_FOR_V403
    )
    assert (
        mod.select_flagged_for_v403(
            mod.DEFAULT_V402_OUTCOMES
            | {
                "s3_acceptance_gate": False,
                "action_efficiency_improves": False,
                "e3_new_levels_positive": False,
            }
        )
        == mod.PAPO_DIAGNOSTIC_FLAGGED_FOR_V403
    )


def test_extract_v402_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4354: source artifacts determine the condition."""

    outcomes = mod.extract_v402_outcomes(
        s3={
            "honest_verdict": "controls_not_differentiable",
            "acceptance_gate": True,
            "benchmark_n": 240,
            "adversarial_verify": {
                "stdout_tail": "[CRITICAL] TAUTOLOGY s3_minus_best_of_k_delta"
            },
        },
        e3={
            "honest_verdict": "success_e3_deeper_tn36_reproduced",
            "new_levels_reproduced": 1,
            "reproducible_total_levels": 23,
            "verifier_is_oracle": True,
            "per_target_scorecard": [
                {
                    "game": "tn36",
                    "offline_reproduced": True,
                    "checkpoint_status": "new_level_reproduced",
                }
            ],
        },
        action={
            "honest_verdict": "success: learned_action_cost_reduces_actions_25_to_16",
            "action_efficiency_improves": True,
            "held_out_actions_baseline": 25,
            "held_out_actions_learned": 16,
            "positive_control_passed": True,
            "reproduction_gated": True,
            "verifier_is_oracle": False,
        },
    )

    assert outcomes == mod.DEFAULT_V402_OUTCOMES


def test_outcome_helpers_reject_malformed_inputs_for_req_report_4354() -> None:
    """REQ-REPORT-4354: malformed source fields never imply a strong branch."""

    assert mod._adversarial_tautology_flagged({"stdout_tail": "clean"}) is False
    assert mod._adversarial_tautology_flagged("not-a-dict") is False
    assert mod._tn36_new_level_reproduced({"per_target_scorecard": 1}) is False
    assert mod._tn36_new_level_reproduced({"per_target_scorecard": "bad"}) is False
    assert mod._tn36_new_level_reproduced({"per_target_scorecard": []}) is False


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
                        "v402_outcome_conditioning": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2602.01842"}
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
        (_valid_artifact() | {"flagged_for_v403": ""}, "flagged_for_v403"),
        (
            _valid_artifact() | {"flagged_for_v403": "unconditioned_followup_v403"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4354"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4354(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4354: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4354: artifact fields are exact."""

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
    """REQ-REPORT-4354: A2D2/SEPO out-of-band rows are source-gated."""

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


def test_validate_studying_section_checks_scenario_report_4354_content() -> None:
    """SCENARIO-REPORT-4354: studying entry maps sources to .403 targets."""

    section = """
    ## 2026-06-17 Exp 4354 - .402 fork SOTA ingestion ingested
    network precondition passed sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch /deep-research not invoked
    acceptance_gate=true honest_verdict=controls_not_differentiable benchmark_n=240 TAUTOLOGY
    success_e3_deeper_tn36_reproduced new_levels_reproduced=1 reproducible_total_levels=23 verifier_is_oracle=true
    action_efficiency_improves=true held_out_actions_baseline=25 held_out_actions_learned=16 positive_control_passed=true reproduction_gated=true verifier_is_oracle=false
    arXiv:2602.01842 arXiv:2604.06260 arXiv:2606.08501 arXiv:2605.05138 arXiv:2503.18809
    out_of_band_flagged arXiv:2606.13565 arXiv:2502.01384 operator-owned NOT auto-run
    flagged_for_v403: prism_hardened_s3_verifier_guided_search_v403
    random_seed=4354
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4354: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v403"):
        mod.validate_studying_section("## Fresh pass\narXiv:2602.01842\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2602.01842", "Prism")
        )

    with pytest.raises(ValueError, match="out_of_band_flagged"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("out_of_band_flagged", "operator note")
        )

    with pytest.raises(ValueError, match="out-of-band citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2606.13565", "A2D2")
        )

    with pytest.raises(ValueError, match="action_efficiency_improves=true"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "action_efficiency_improves=true",
                "action_efficiency_improves=false",
            )
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4354(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4354: writer emits artifact and studying entry."""

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
    assert studying.count("2026-06-17 Exp 4354") == 1
    assert "flagged_for_v403" in studying
    assert "out_of_band_flagged" in studying
    assert "acceptance_gate=true" in studying
    assert "action_efficiency_improves=true" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4354() -> None:
    """REQ-REPORT-4354: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-17 Exp 4354") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-17 Exp 4354")
    assert studying_refreshed.count("2026-06-17 Exp 4354") == 1
    assert marker_at_end.count("2026-06-17 Exp 4354") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "Prism" in no_heading


def test_main_prints_terminal_verdict_for_req_report_4354(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4354: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4354(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4354: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4354_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4354(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4354: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4354_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4354() -> None:
    """REQ-REPORT-4354: committed JSON artifact satisfies the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v403"] == mod.DEFAULT_FLAGGED_FOR_V403
    assert artifact["out_of_band_flagged"] == mod.DEFAULT_OUT_OF_BAND_FLAGGED
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-17 Exp 4354 - .402 fork SOTA ingestion ingested" in studying
    assert (
        "Flagged for .403: `prism_hardened_s3_verifier_guided_search_v403`"
    ) in studying
