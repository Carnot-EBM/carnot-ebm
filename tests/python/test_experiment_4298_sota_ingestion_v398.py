"""Tests for REQ-REPORT-4298 / SCENARIO-REPORT-4298."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4298_sota_ingestion_v398 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4298_sota_ingestion_v398.json")
WRAPPER_PATH = Path("results/experiment_4298_sota_ingestion_v398.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v398=mod.DEFAULT_FLAGGED_FOR_V398,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4298_spec_anchor_exists() -> None:
    """REQ-REPORT-4298: OpenSpec declares the .398 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4298" in spec
    assert "SCENARIO-REPORT-4298" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v398" in spec
    assert "cross_generator_holds=true" in spec
    assert "partial_state_leak_free=true" in spec
    assert "partial_state_auroc=0.966143" in spec
    assert "Exp 4294 JSON is absent" in spec
    for source in mod.VERIFIED_SOURCE_URLS.values():
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4298() -> None:
    """REQ-REPORT-4298: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v398": mod.DEFAULT_FLAGGED_FOR_V398,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL (no citation = "
                "fabrication per adversarial_verify discipline) + a one-line "
                ".398 experiment mapping."
            ),
            "flagged_for_v398": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the .398 planner, conditioned on the .397 outcomes."
            ),
            "random_seed": (
                "Determinism placeholder for the discovery query set (recorded "
                "for reproducibility of the sweep)."
            ),
        },
    }


def test_select_flagged_for_v398_conditions_on_v397_outcomes() -> None:
    """SCENARIO-REPORT-4298: .398 flag changes with the .397 outcomes."""

    assert mod.select_flagged_for_v398(mod.DEFAULT_V397_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V398
    assert (
        mod.select_flagged_for_v398(
            mod.DEFAULT_V397_OUTCOMES
            | {"partial_state_scorer_built": False, "partial_state_leak_free": False}
        )
        == mod.NONLEAKY_SCORER_REPAIR_FLAGGED_FOR_V398
    )
    assert (
        mod.select_flagged_for_v398(
            mod.DEFAULT_V397_OUTCOMES
            | {
                "cross_generator_holds": False,
                "non_degenerate_guards_pass": False,
                "partial_state_scorer_built": True,
                "partial_state_leak_free": True,
            }
        )
        == mod.STRONG_GUIDED_GENERATION_FLAGGED_FOR_V398
    )
    assert (
        mod.select_flagged_for_v398(
            mod.DEFAULT_V397_OUTCOMES
            | {
                "cross_generator_holds": False,
                "partial_state_scorer_built": True,
                "partial_state_leak_free": True,
                "efficiency_pareto_holds": True,
                "strong_judge_artifact_available": True,
            }
        )
        == mod.SMALL_VERIFIER_EFFICIENCY_FLAGGED_FOR_V398
    )


def test_extract_v397_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4298: source artifacts determine the actual condition."""

    outcomes = mod.extract_v397_outcomes(
        arcgen={
            "cross_generator_holds": True,
            "non_degenerate_guards_pass": True,
            "headline_outcome": "arcgen_cross_generator_generalizes",
        },
        partial_state={
            "partial_state_scorer_built": True,
            "partial_state_leak_free": True,
            "partial_state_auroc": 0.966143,
            "leak_ablation_auroc": 0.937365,
        },
        efficiency=None,
    )

    assert outcomes == mod.DEFAULT_V397_OUTCOMES


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
                        "v397_outcome_conditioning": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2606.11182"}
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
        (_valid_artifact() | {"flagged_for_v398": ""}, "flagged_for_v398"),
        (
            _valid_artifact() | {"flagged_for_v398": "unconditioned_followup_v398"},
            "conditioned",
        ),
        (_valid_artifact() | {"random_seed": "4298"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4298(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4298: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_fields() -> None:
    """SCENARIO-REPORT-4298: artifact fields are exact."""

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


def test_validate_studying_section_checks_scenario_report_4298_content() -> None:
    """SCENARIO-REPORT-4298: studying entry maps sources to .398 targets."""

    section = """
    ## 2026-06-16 Exp 4298 - .397 fork SOTA ingestion ingested
    sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch /deep-research not invoked
    cross_generator_holds=true non_degenerate_guards_pass=true arcgen_cross_generator_generalizes
    partial_state_scorer_built=true partial_state_leak_free=true partial_state_auroc=0.966143 leak_ablation_auroc=0.937365
    strong_judge_efficiency_outcome=unavailable_missing_exp4294_json
    arXiv:2606.11182 arXiv:2605.19633 arXiv:2606.14211 arXiv:2606.07810 arXiv:2503.00307
    flagged_for_v398: eevee_router_prompt_broader_domain_selector_v398
    random_seed=4298
    """

    mod.validate_studying_section(section)


def test_validate_studying_section_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4298: studying entry must cite sources and close the loop."""

    with pytest.raises(ValueError, match="flagged_for_v398"):
        mod.validate_studying_section("## Fresh-pass provenance\narXiv:2606.11182\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("arXiv:2606.11182", "EEVEE")
        )

    with pytest.raises(ValueError, match="cross_generator_holds=true"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("cross_generator_holds=true", "cross_generator=false")
        )

    with pytest.raises(ValueError, match="partial_state_leak_free=true"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("partial_state_leak_free=true", "partial_state_leak_free=false")
        )

    with pytest.raises(ValueError, match="unavailable_missing_exp4294_json"):
        mod.validate_studying_section(
            mod.STUDYING_SECTION.replace("unavailable_missing_exp4294_json", "complete")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4298(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4298: writer emits artifact and studying entry."""

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
    assert studying.count("2026-06-16 Exp 4298") == 1
    assert "flagged_for_v398" in studying
    assert "cross_generator_holds=true" in studying
    assert "partial_state_leak_free=true" in studying
    assert "unavailable_missing_exp4294_json" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4298() -> None:
    """REQ-REPORT-4298: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-16 Exp 4298") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-16 Exp 4298")
    assert studying_refreshed.count("2026-06-16 Exp 4298") == 1
    assert marker_at_end.count("2026-06-16 Exp 4298") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "treat small-verifier efficiency as\nunconfirmed until Exp 4294 exists." in no_heading


def test_main_prints_terminal_verdict_for_req_report_4298(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4298: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4298(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4298: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4298_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4298(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4298: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4298_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4298() -> None:
    """REQ-REPORT-4298: committed JSON artifact satisfies the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v398"] == mod.DEFAULT_FLAGGED_FOR_V398
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-16 Exp 4298 - .397 fork SOTA ingestion ingested" in studying
    assert "Flagged for .398: `eevee_router_prompt_broader_domain_selector_v398`" in studying
