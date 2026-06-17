"""Tests for REQ-REPORT-4343 / SCENARIO-REPORT-4343."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4343_sota_ingestion_v402 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4343_sota_ingestion_v402.json")
WRAPPER_PATH = Path("results/experiment_4343_sota_ingestion_v402.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v402=mod.DEFAULT_FLAGGED_FOR_V402,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4343_spec_anchor_exists() -> None:
    """REQ-REPORT-4343: OpenSpec declares the .402 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4343" in spec
    assert "SCENARIO-REPORT-4343" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v402" in spec
    assert "in_generation_moat_replicates=true" in spec
    assert "offline_reproduced=true" in spec
    assert "learned_encoder_transfer_helps=false" in spec
    for source in mod.VERIFIED_SOURCE_URLS.values():
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4343() -> None:
    """REQ-REPORT-4343: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v402": mod.DEFAULT_FLAGGED_FOR_V402,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable "
                "citations (or blocked_network_unavailable)."
            ),
            "methods_mapped": (
                "Each method MUST carry a real, VERIFIED arXiv ID/URL (no "
                "citation = fabrication) + a one-line .402 experiment mapping "
                "+ the failure mode + the .401-outcome conditioning."
            ),
            "flagged_for_v402": (
                "Closes discover->ingest->plan: names the single strongest "
                "method for the .402 planner, conditioned on the .401 "
                "in-generation-moat-settle + E3 + self-learning outcomes."
            ),
            "random_seed": (
                "Determinism placeholder for the discovery query set "
                "(reproducibility of the sweep)."
            ),
        },
    }


def test_select_flagged_for_v402_conditions_on_v401_outcomes() -> None:
    """SCENARIO-REPORT-4343: .402 flag follows the .401 branch decision."""

    assert mod.select_flagged_for_v402(mod.DEFAULT_V401_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V402
    assert (
        mod.select_flagged_for_v402(
            mod.DEFAULT_V401_OUTCOMES
            | {
                "in_generation_moat_replicates": False,
                "scorer_leak_recheck_passed": False,
                "replication_ci_excludes_zero": False,
            }
        )
        == mod.CONSEQUENCE_ORACLE_FREE_FLAGGED_FOR_V402
    )
    assert (
        mod.select_flagged_for_v402(
            mod.DEFAULT_V401_OUTCOMES
            | {
                "in_generation_moat_replicates": True,
                "scorer_leak_recheck_passed": False,
            }
        )
        == mod.CONSEQUENCE_ORACLE_FREE_FLAGGED_FOR_V402
    )
    assert (
        mod.select_flagged_for_v402(
            mod.DEFAULT_V401_OUTCOMES
            | {
                "in_generation_moat_replicates": True,
                "scorer_leak_recheck_passed": True,
                "replication_ci_excludes_zero": False,
                "e3_ar25_reproduced": True,
                "e3_sc25_reproduced": True,
            }
        )
        == mod.MULTI_GAME_E3_FLAGGED_FOR_V402
    )
    assert (
        mod.select_flagged_for_v402(
            mod.DEFAULT_V401_OUTCOMES
            | {
                "in_generation_moat_replicates": True,
                "scorer_leak_recheck_passed": True,
                "replication_ci_excludes_zero": False,
                "e3_ar25_reproduced": False,
                "e3_sc25_reproduced": False,
                "learned_encoder_transfer_helps": True,
            }
        )
        == mod.WORLD_MODEL_INTERACTION_FLAGGED_FOR_V402
    )
    assert (
        mod.select_flagged_for_v402(
            mod.DEFAULT_V401_OUTCOMES
            | {
                "in_generation_moat_replicates": True,
                "scorer_leak_recheck_passed": True,
                "replication_ci_excludes_zero": False,
                "e3_ar25_reproduced": False,
                "e3_sc25_reproduced": False,
                "learned_encoder_transfer_helps": False,
            }
        )
        == mod.CONSEQUENCE_ORACLE_FREE_FLAGGED_FOR_V402
    )


def test_extract_v401_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4343: source artifacts determine the actual condition."""

    outcomes = mod.extract_v401_outcomes(
        moat={
            "honest_verdict": "complete: in_generation_moat_replicates",
            "in_generation_moat_replicates": True,
            "scorer_leak_recheck_passed": True,
            "controls_differentiated": True,
            "benchmark_n": 240,
            "carnot_minus_best_control_delta": 0.358333,
            "replication_ci95": [0.283333, 0.4375],
        },
        e3_ar25={
            "offline_reproduced": True,
            "plan_executed": True,
            "reproduced_levels": 1,
            "game": "ar25",
            "explore_lemmas_collected": 7,
        },
        e3_sc25={
            "offline_reproduced": True,
            "plan_executed": True,
            "reproduced_levels": 1,
            "game": "sc25",
            "explore_lemmas_collected": 6,
        },
        transfer={
            "learned_encoder_transfer_helps": False,
            "cross_game_state_reduction": 1.00635593220339,
            "cross_game_state_reduction_ci95": [1.0, 1.0168354897287482],
            "positive_control_passed": True,
        },
    )

    assert outcomes == mod.DEFAULT_V401_OUTCOMES


def test_ci_helpers_reject_malformed_inputs_for_req_report_4343() -> None:
    """REQ-REPORT-4343: malformed intervals never imply a decision-grade win."""

    assert mod._ci_excludes_zero("not-a-ci") is False
    assert mod._ci_excludes_zero([-1.0, 1.0]) is False
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
                        "source_verification": "fake",
                        "track": "fake",
                        "v401_outcome_conditioning": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2604.06260"}
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
        (_valid_artifact() | {"flagged_for_v402": ""}, "flagged_for_v402"),
        (
            _valid_artifact() | {"flagged_for_v402": "unconditioned_followup_v402"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4343"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4343(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4343: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4343: artifact fields are exact."""

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


def test_validate_studying_section_checks_scenario_report_4343_content() -> None:
    """SCENARIO-REPORT-4343: studying entry maps sources to .402 targets."""

    section = """
    ## 2026-06-17 Exp 4343 - .401 outcome SOTA ingestion ingested
    network precondition passed sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch /deep-research not invoked
    in_generation_moat_replicates=true scorer_leak_recheck_passed=true controls_differentiated=true benchmark_n=240 carnot_minus_best_control_delta=0.358333 replication_ci95=[0.283333, 0.4375]
    game=ar25 offline_reproduced=true plan_executed=true reproduced_levels=1 explore_lemmas_collected=7
    game=sc25 offline_reproduced=true plan_executed=true reproduced_levels=1 explore_lemmas_collected=6
    learned_encoder_transfer_helps=false cross_game_state_reduction=1.00635593220339 cross_game_state_reduction_ci95=[1.0, 1.0168354897287482] positive_control_passed=true
    arXiv:2604.06260 arXiv:2606.13565 arXiv:2606.08501 arXiv:2605.05138 arXiv:2605.15256
    flagged_for_v402: s3_stratified_scaling_search_guided_generation_v402
    random_seed=4343
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4343: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v402"):
        mod.validate_studying_section("## Fresh pass\narXiv:2604.06260\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2604.06260", "S3")
        )

    with pytest.raises(ValueError, match="in_generation_moat_replicates=true"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "in_generation_moat_replicates=true",
                "in_generation_moat_replicates=false",
            )
        )

    with pytest.raises(ValueError, match="learned_encoder_transfer_helps=false"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "learned_encoder_transfer_helps=false",
                "learned_encoder_transfer_helps=true",
            )
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4343(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4343: writer emits artifact and studying entry."""

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
    assert studying.count("2026-06-17 Exp 4343") == 1
    assert "flagged_for_v402" in studying
    assert "in_generation_moat_replicates=true" in studying
    assert "offline_reproduced=true" in studying
    assert "learned_encoder_transfer_helps=false" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4343() -> None:
    """REQ-REPORT-4343: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-17 Exp 4343") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-17 Exp 4343")
    assert studying_refreshed.count("2026-06-17 Exp 4343") == 1
    assert marker_at_end.count("2026-06-17 Exp 4343") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "Stratified Scaling Search" in no_heading


def test_main_prints_terminal_verdict_for_req_report_4343(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4343: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4343(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4343: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4343_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4343(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4343: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4343_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4343() -> None:
    """REQ-REPORT-4343: committed JSON artifact satisfies the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v402"] == mod.DEFAULT_FLAGGED_FOR_V402
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-17 Exp 4343 - .401 outcome SOTA ingestion ingested" in studying
    assert (
        "Flagged for .402: "
        "`s3_stratified_scaling_search_guided_generation_v402`"
    ) in studying
