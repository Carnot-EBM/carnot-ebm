"""Tests for REQ-REPORT-4309 / SCENARIO-REPORT-4309."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4309_sota_ingestion_v399 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4309_sota_ingestion_v399.json")
WRAPPER_PATH = Path("results/experiment_4309_sota_ingestion_v399.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v399=mod.DEFAULT_FLAGGED_FOR_V399,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4309_spec_anchor_exists() -> None:
    """REQ-REPORT-4309: OpenSpec declares the .399 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4309" in spec
    assert "SCENARIO-REPORT-4309" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v399" in spec
    assert "efficiency_pareto_holds=true" in spec
    assert "diffusiongemma_guidance_moat=false" in spec
    assert "cross_domain_selection_holds=false" in spec
    assert "cross_domain_delta=0.2307692308" in spec
    for source in mod.VERIFIED_SOURCE_URLS.values():
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4309() -> None:
    """REQ-REPORT-4309: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v399": mod.DEFAULT_FLAGGED_FOR_V399,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL (no citation = "
                "fabrication per adversarial_verify discipline) + a one-line "
                ".399 experiment mapping."
            ),
            "flagged_for_v399": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the .399 planner, conditioned on the .398 outcomes."
            ),
            "random_seed": (
                "Determinism placeholder for the discovery query set (recorded "
                "for reproducibility of the sweep)."
            ),
        },
    }


def test_select_flagged_for_v399_conditions_on_v398_outcomes() -> None:
    """SCENARIO-REPORT-4309: .399 flag changes with the .398 outcomes."""

    assert mod.select_flagged_for_v399(mod.DEFAULT_V398_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V399
    assert (
        mod.select_flagged_for_v399(
            mod.DEFAULT_V398_OUTCOMES
            | {
                "efficiency_pareto_holds": False,
                "diffusiongemma_guidance_moat": True,
                "controls_differentiated": True,
                "scorer_leak_recheck_passed": True,
            }
        )
        == mod.SMC_GUIDED_GENERATION_FLAGGED_FOR_V399
    )
    assert (
        mod.select_flagged_for_v399(
            mod.DEFAULT_V398_OUTCOMES
            | {
                "efficiency_pareto_holds": False,
                "diffusiongemma_guidance_moat": False,
                "cross_domain_selection_holds": True,
                "label_ablation_robust": True,
            }
        )
        == mod.FOURTH_DOMAIN_ROUTER_FLAGGED_FOR_V399
    )
    assert (
        mod.select_flagged_for_v399(
            mod.DEFAULT_V398_OUTCOMES
            | {
                "efficiency_pareto_holds": False,
                "diffusiongemma_guidance_moat": False,
                "cross_domain_selection_holds": False,
            }
        )
        == mod.ROUTER_REBUILD_FLAGGED_FOR_V399
    )


def test_extract_v398_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4309: source artifacts determine the actual condition."""

    outcomes = mod.extract_v398_outcomes(
        efficiency={
            "efficiency_pareto_holds": True,
            "accuracy_energy_verifier": 0.8,
            "accuracy_best_judge": 0.5,
            "cost_ratio": 1.03e-08,
        },
        guidance={
            "diffusiongemma_guidance_moat": False,
            "controls_differentiated": True,
            "scorer_leak_recheck_passed": True,
            "carnot_minus_best_control_delta": 0.133334,
        },
        cross_domain={
            "cross_domain_selection_holds": False,
            "cross_domain_delta": 0.2307692308,
            "cross_domain_ci95": [-0.1153846154, 0.5384615385],
            "label_ablation_robust": True,
        },
    )

    assert outcomes == mod.DEFAULT_V398_OUTCOMES


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
                        "v398_outcome_conditioning": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2510.14913"}
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
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
        (_valid_artifact() | {"flagged_for_v399": ""}, "flagged_for_v399"),
        (
            _valid_artifact() | {"flagged_for_v399": "unconditioned_followup_v399"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4309"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4309(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4309: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4309: artifact fields are exact."""

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


def test_validate_studying_section_checks_scenario_report_4309_content() -> None:
    """SCENARIO-REPORT-4309: studying entry maps sources to .399 targets."""

    section = """
    ## 2026-06-17 Exp 4309 - .398 fork SOTA ingestion ingested
    sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch /deep-research not invoked
    efficiency_pareto_holds=true accuracy_energy_verifier=0.8 accuracy_best_judge=0.5 cost_ratio=1.03e-08
    diffusiongemma_guidance_moat=false controls_differentiated=true scorer_leak_recheck_passed=true
    cross_domain_selection_holds=false cross_domain_delta=0.2307692308 cross_domain_ci95=[-0.1153846154, 0.5384615385]
    arXiv:2510.14913 arXiv:2606.06098 arXiv:2601.09692 arXiv:2601.11443 arXiv:2505.22524
    flagged_for_v399: budget_aware_discriminative_cascade_router_v399
    random_seed=4309
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4309: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v399"):
        mod.validate_studying_section("## Fresh-pass provenance\narXiv:2510.14913\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2510.14913", "Budget-aware")
        )

    with pytest.raises(ValueError, match="efficiency_pareto_holds=true"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("efficiency_pareto_holds=true", "efficiency=false")
        )

    with pytest.raises(ValueError, match="diffusiongemma_guidance_moat=false"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "diffusiongemma_guidance_moat=false",
                "diffusiongemma_guidance_moat=true",
            )
        )

    with pytest.raises(ValueError, match="cross_domain_selection_holds=false"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace(
                "cross_domain_selection_holds=false",
                "cross_domain_selection_holds=true",
            )
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4309(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4309: writer emits artifact and studying entry."""

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
    assert studying.count("2026-06-17 Exp 4309") == 1
    assert "flagged_for_v399" in studying
    assert "efficiency_pareto_holds=true" in studying
    assert "diffusiongemma_guidance_moat=false" in studying
    assert "cross_domain_selection_holds=false" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4309() -> None:
    """REQ-REPORT-4309: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-17 Exp 4309") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-17 Exp 4309")
    assert studying_refreshed.count("2026-06-17 Exp 4309") == 1
    assert marker_at_end.count("2026-06-17 Exp 4309") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "keep SMC-guided DiffusionGemma as the secondary repair track" in no_heading


def test_main_prints_terminal_verdict_for_req_report_4309(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4309: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4309(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4309: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4309_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4309(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4309: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4309_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4309() -> None:
    """REQ-REPORT-4309: committed JSON artifact satisfies the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v399"] == mod.DEFAULT_FLAGGED_FOR_V399
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-17 Exp 4309 - .398 fork SOTA ingestion ingested" in studying
    assert "Flagged for .399: `budget_aware_discriminative_cascade_router_v399`" in studying
