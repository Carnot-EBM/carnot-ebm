"""Tests for REQ-REPORT-4332 / SCENARIO-REPORT-4332."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4332_sota_ingestion_v401 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4332_sota_ingestion_v401.json")
WRAPPER_PATH = Path("results/experiment_4332_sota_ingestion_v401.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v401=mod.DEFAULT_FLAGGED_FOR_V401,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4332_spec_anchor_exists() -> None:
    """REQ-REPORT-4332: OpenSpec declares the .401 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4332" in spec
    assert "SCENARIO-REPORT-4332" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v401" in spec
    assert "in_generation_moat_replicates=false" in spec
    assert "adaptive_guidance_beats_control=false" in spec
    assert "offline_reproduced=false" in spec
    assert "learned_encoder_transfer_helps=false" in spec
    for source in mod.VERIFIED_SOURCE_URLS.values():
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4332() -> None:
    """REQ-REPORT-4332: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v401": mod.DEFAULT_FLAGGED_FOR_V401,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real, VERIFIED arXiv ID/URL (no "
                "citation = fabrication per adversarial_verify discipline) + "
                "a one-line .401 experiment mapping."
            ),
            "flagged_for_v401": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the .401 planner, conditioned on the .400 outcomes."
            ),
            "random_seed": (
                "Determinism placeholder for the discovery query set (recorded "
                "for reproducibility of the sweep)."
            ),
        },
    }


def test_select_flagged_for_v401_conditions_on_v400_outcomes() -> None:
    """SCENARIO-REPORT-4332: .401 flag changes with the .400 outcomes."""

    assert mod.select_flagged_for_v401(mod.DEFAULT_V400_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V401
    assert (
        mod.select_flagged_for_v401(
            mod.DEFAULT_V400_OUTCOMES
            | {
                "in_generation_moat_replicates": True,
                "adaptive_guidance_beats_control": True,
                "scorer_leak_recheck_passed": True,
            }
        )
        == mod.SCALED_ARC_GRID_GENERATION_FLAGGED_FOR_V401
    )
    assert (
        mod.select_flagged_for_v401(
            mod.DEFAULT_V400_OUTCOMES
            | {
                "scorer_leak_recheck_passed": True,
                "offline_reproduced": True,
                "reproduced_levels_positive": True,
            }
        )
        == mod.MULTI_GAME_E3_SWEEP_FLAGGED_FOR_V401
    )
    assert (
        mod.select_flagged_for_v401(
            mod.DEFAULT_V400_OUTCOMES
            | {
                "scorer_leak_recheck_passed": True,
                "learned_encoder_transfer_helps": True,
                "baseline_solves_held_out": True,
            }
        )
        == mod.RICHER_ENCODER_MORE_GAMES_FLAGGED_FOR_V401
    )
    assert (
        mod.select_flagged_for_v401(
            mod.DEFAULT_V400_OUTCOMES
            | {
                "scorer_leak_recheck_passed": True,
                "adaptive_guidance_beats_control": True,
            }
        )
        == mod.REWARD_SCORE_MATCHING_ABLATION_FLAGGED_FOR_V401
    )
    assert (
        mod.select_flagged_for_v401(
            mod.DEFAULT_V400_OUTCOMES | {"scorer_leak_recheck_passed": True}
        )
        == mod.AGENT2WORLD_E3_REPAIR_FLAGGED_FOR_V401
    )


def test_extract_v400_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4332: source artifacts determine the actual condition."""

    outcomes = mod.extract_v400_outcomes(
        moat={
            "honest_verdict": "scorer_leaky_on_second_corpus",
            "in_generation_moat_replicates": False,
            "controls_differentiated": False,
            "scorer_leak_recheck_passed": False,
            "benchmark_n": 0,
            "carnot_minus_best_control_delta": 0.0,
            "replication_ci95": [0.0, 0.0],
        },
        adaptive={
            "adaptive_guidance_beats_control": False,
            "adaptive_ci95": [-0.075, 0.35],
            "controls_differentiated": True,
            "benchmark_n": 40,
        },
        e3={
            "offline_reproduced": False,
            "plan_executed": False,
            "reproduced_levels": 0,
            "verifier_best_accuracy": 0.8875,
            "residual_mismatch_class": "missing_world_model_rule_gap_hidden_undo_stack_action7",
        },
        transfer={
            "learned_encoder_transfer_helps": False,
            "cross_game_state_reduction": 1.0084925690021231,
            "cross_game_state_reduction_ci95": [1.0, 1.0303068758652514],
            "baseline_solves_held_out": True,
        },
    )

    assert outcomes == mod.DEFAULT_V400_OUTCOMES


def test_ci_helpers_reject_malformed_inputs_for_req_report_4332() -> None:
    """REQ-REPORT-4332: malformed confidence intervals never imply a win."""

    assert mod._ci_excludes_zero("not-a-ci") is False
    assert mod._ci_lower_exceeds_one([1.1]) is False


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
                        "track": "fake",
                        "source_read": "fake",
                        "v400_outcome_conditioning": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2602.11146"}
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
        (_valid_artifact() | {"flagged_for_v401": ""}, "flagged_for_v401"),
        (
            _valid_artifact() | {"flagged_for_v401": "unconditioned_followup_v401"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4332"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4332(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4332: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4332: artifact fields are exact."""

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


def test_validate_studying_section_checks_scenario_report_4332_content() -> None:
    """SCENARIO-REPORT-4332: studying entry maps sources to .401 targets."""

    section = """
    ## 2026-06-17 Exp 4332 - .400 fork SOTA ingestion ingested
    sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch /deep-research not invoked
    in_generation_moat_replicates=false controls_differentiated=false scorer_leak_recheck_passed=false benchmark_n=0 carnot_minus_best_control_delta=0.0 replication_ci95=[0.0, 0.0]
    adaptive_guidance_beats_control=false adaptive_ci95=[-0.075, 0.35] adaptive_controls_differentiated=true adaptive_benchmark_n=40
    offline_reproduced=false plan_executed=false reproduced_levels=0 verifier_best_accuracy=0.8875 residual_mismatch_class=missing_world_model_rule_gap_hidden_undo_stack_action7
    learned_encoder_transfer_helps=false cross_game_state_reduction=1.0084925690021231 cross_game_state_reduction_ci95=[1.0, 1.0303068758652514] baseline_solves_held_out=true
    arXiv:2602.11146 arXiv:2502.01384 arXiv:2512.22336 arXiv:2605.25931 arXiv:2605.15256
    flagged_for_v401: leak_robust_diffusion_native_partial_state_reward_v401
    random_seed=4332
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4332: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v401"):
        mod.validate_studying_section("## Fresh-pass provenance\narXiv:2602.11146\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2602.11146", "DiNa-LRM")
        )

    with pytest.raises(ValueError, match="in_generation_moat_replicates=false"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "in_generation_moat_replicates=false",
                "in_generation_moat_replicates=true",
            )
        )

    with pytest.raises(ValueError, match="offline_reproduced=false"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "offline_reproduced=false",
                "offline_reproduced=true",
            )
        )

    with pytest.raises(ValueError, match="learned_encoder_transfer_helps=false"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "learned_encoder_transfer_helps=false",
                "learned_encoder_transfer_helps=true",
            )
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4332(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4332: writer emits artifact and studying entry."""

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
    assert studying.count("2026-06-17 Exp 4332") == 1
    assert "flagged_for_v401" in studying
    assert "in_generation_moat_replicates=false" in studying
    assert "offline_reproduced=false" in studying
    assert "learned_encoder_transfer_helps=false" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4332() -> None:
    """REQ-REPORT-4332: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-17 Exp 4332") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-17 Exp 4332")
    assert studying_refreshed.count("2026-06-17 Exp 4332") == 1
    assert marker_at_end.count("2026-06-17 Exp 4332") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "leak-robust diffusion-native partial-state reward" in no_heading


def test_main_prints_terminal_verdict_for_req_report_4332(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4332: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4332(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4332: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4332_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4332(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4332: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4332_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4332() -> None:
    """REQ-REPORT-4332: committed JSON artifact satisfies the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v401"] == mod.DEFAULT_FLAGGED_FOR_V401
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-17 Exp 4332 - .400 fork SOTA ingestion ingested" in studying
    assert (
        "Flagged for .401: "
        "`leak_robust_diffusion_native_partial_state_reward_v401`"
    ) in studying
