"""Tests for REQ-REPORT-4429 / SCENARIO-REPORT-4429."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4429_sota_ingestion_409 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4429_sota_ingestion_409.json")
WRAPPER_PATH = Path("results/experiment_4429_sota_ingestion_409.py")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v410=mod.DEFAULT_FLAGGED_FOR_V410,
        outcome_conditioning=dict(mod.DEFAULT_V409_OUTCOMES),
        preconditions_checked=dict(mod.DEFAULT_PRECONDITIONS_CHECKED),
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4429_spec_anchor_exists() -> None:
    """REQ-REPORT-4429: OpenSpec declares the .409 headline ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4429" in spec
    assert "SCENARIO-REPORT-4429" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v410" in spec
    assert "inference_substrate" in spec
    assert "cpu_reliable_channel_sota_ingestion_no_training" in spec
    assert "arXiv:2605.05138" in spec


def test_build_artifact_has_required_fields_for_req_report_4429() -> None:
    """REQ-REPORT-4429: artifact exposes the exact required fields."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["flagged_for_v410"] == mod.DEFAULT_FLAGGED_FOR_V410
    assert artifact["methods_mapped"] == _valid_methods()
    assert artifact["outcome_conditioning"] == mod.DEFAULT_V409_OUTCOMES
    assert artifact["preconditions_checked"] == mod.DEFAULT_PRECONDITIONS_CHECKED
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert "SOTA->experiment" in artifact["sota_to_experiment_mapping_note"]


def test_extract_v409_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4429: .409 source artifacts determine the .410 branch."""

    outcomes = mod.extract_v409_outcomes(
        config_rule={
            "honest_verdict": "success_s5i5_L1_offline_reproduced",
            "new_levels_reproduced": 1,
            "offline_reproduced": True,
            "verifier_is_oracle": True,
            "flagged_adversarial": True,
        },
        first_contact={
            "honest_verdict": "partial: generic_first_contact_g50t_routed_missing_verifier_gap_logged",
            "target_game": "g50t",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "missing_verifier_gaps": [{"gap_id": "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT"}],
            "verifier_is_oracle": False,
        },
        deeper_world_model={
            "honest_verdict": "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap",
            "new_levels_reproduced": 0,
            "offline_reproduced": False,
            "per_mechanic_test_pass_rate": 0.5,
            "residual_failing_mechanic": "sc25_l2_route_search_still_missing_after_hud_cleanup",
            "verifier_is_oracle": True,
        },
        vocabulary_transfer={
            "honest_verdict": "complete: null_config_rule_vocabulary_transfer_seeded_arm_missing",
            "config_rule_vocabulary_transfers": False,
            "flagged_adversarial": True,
            "verifier_is_oracle": False,
        },
        registry_audit={
            "honest_verdict": "complete: registry_repro_audit_total_35_asserted_36_audited",
            "inference_substrate": "offline_arc_registry_repro_audit_cpu_no_llm",
            "all_counted_entries_reproduced": True,
            "registry_claimed_reproducible_total_levels": 35,
            "counted_entries_audited": 18,
            "milestone_409_reproduction_gates": [
                {"experiment": "exp4421", "new_levels_counted": 1, "reproduction_gated": True},
                {"experiment": "exp4423", "new_levels_counted": 0, "reproduction_gated": True},
                {"experiment": "exp4424", "new_levels_counted": 0, "reproduction_gated": True},
            ],
        },
    )

    assert outcomes == mod.DEFAULT_V409_OUTCOMES


def test_select_flagged_for_v410_conditions_on_v409_outcomes() -> None:
    """SCENARIO-REPORT-4429: .410 flag follows the .409 branch decision."""

    assert mod.select_flagged_for_v410(mod.DEFAULT_V409_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V410
    assert (
        mod.select_flagged_for_v410(
            mod.DEFAULT_V409_OUTCOMES
            | {
                "generic_first_contact_partial": False,
                "deeper_world_model_no_new_level": False,
                "config_rule_level_counted_after_gate": True,
            }
        )
        == mod.REACOMP_FLAGGED_FOR_V410
    )
    assert (
        mod.select_flagged_for_v410(
            mod.DEFAULT_V409_OUTCOMES
                | {
                    "generic_first_contact_partial": True,
                    "deeper_world_model_no_new_level": False,
                    "registry_all_counted_entries_reproduced": False,
                    "config_rule_level_counted_after_gate": False,
                }
            )
            == mod.AERA_FLAGGED_FOR_V410
    )
    assert (
        mod.select_flagged_for_v410(
            mod.DEFAULT_V409_OUTCOMES
            | {
                "generic_first_contact_partial": False,
                "deeper_world_model_no_new_level": True,
                "config_rule_level_counted_after_gate": False,
            }
        )
        == mod.AGENT2WORLD_FLAGGED_FOR_V410
    )
    assert (
        mod.select_flagged_for_v410(
            mod.DEFAULT_V409_OUTCOMES
            | {
                "generic_first_contact_partial": False,
                "deeper_world_model_no_new_level": False,
                "config_rule_level_counted_after_gate": False,
            }
        )
        == mod.CODEARC_FLAGGED_FOR_V410
    )


def test_outcome_helpers_fail_closed_for_req_report_4429() -> None:
    """REQ-REPORT-4429: malformed optional source fields never imply a strong branch."""

    assert mod._gate_by_experiment([{"experiment": "exp4421"}], "exp4421") == {
        "experiment": "exp4421"
    }
    assert mod._gate_by_experiment("bad", "exp4421") == {}
    assert mod._gate_by_experiment([{"experiment": "other"}], "exp4421") == {}
    assert mod._nonempty_list([1]) is True
    assert mod._nonempty_list([]) is False
    assert mod._nonempty_list("bad") is False

    malformed = mod.extract_v409_outcomes(
        config_rule={"new_levels_reproduced": 1},
        first_contact={"missing_verifier_gaps": "bad"},
        deeper_world_model={"per_mechanic_test_pass_rate": "1.0"},
        vocabulary_transfer={},
        registry_audit={"milestone_409_reproduction_gates": "bad"},
    )
    assert malformed["generic_first_contact_missing_verifier_gap"] is False
    assert malformed["deeper_world_model_mechanic_tests_pass"] is False
    assert malformed["registry_exp4421_new_level_counted"] is False


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"inference_substrate": "gpu_training"}, "CPU"),
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
        (_valid_artifact() | {"flagged_for_v410": ""}, "flagged_for_v410"),
        (_valid_artifact() | {"flagged_for_v410": "unconditioned_followup"}, "conditioned"),
        (_valid_artifact() | {"random_seed": "4429"}, "random_seed"),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"deep_research_invoked": True}},
            "deep-research",
        ),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"trm_training_stood_down": False}},
            "TRM",
        ),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"cpu_only": False}},
            "CPU",
        ),
        (_valid_artifact() | {"outcome_conditioning": {}}, "outcome_conditioning"),
        (_valid_artifact() | {"sota_to_experiment_mapping_note": "too short"}, "mapping note"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4429(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4429: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_method_shape() -> None:
    """REQ-REPORT-4429: artifact fields and method rows are exact."""

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
        "methods_mapped": [_valid_methods()[0] | {"extra": "field"}] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_method_fields)

    bad_url = _valid_artifact() | {
        "methods_mapped": [_valid_methods()[0] | {"url": "https://example.com"}] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="url"):
        mod.validate_artifact(bad_url)

    bad_preconditions = _valid_artifact()
    preconditions = dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
    preconditions.pop("sweep_clusters_ran")
    bad_preconditions["preconditions_checked"] = preconditions
    with pytest.raises(ValueError, match="preconditions_checked"):
        mod.validate_artifact(bad_preconditions)


def test_validate_studying_section_checks_scenario_report_4429_content() -> None:
    """SCENARIO-REPORT-4429: studying entry maps SOTA sources to .410 targets."""

    mod.validate_studying_section(mod.STUDYING_SECTION)

    with pytest.raises(ValueError, match="flagged_for_v410"):
        mod.validate_studying_section("## Fresh pass\narXiv:2605.05138\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("arXiv:2512.22336", "Agent2World"))

    with pytest.raises(ValueError, match="CPU"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("CPU", "accelerated"))


def test_load_v409_outcomes_reads_repo_relative_sources(tmp_path: Path) -> None:
    """REQ-REPORT-4429: loader reads the .409 ARC source artifacts."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payloads = {
        "experiment_4421_config_rule_solve_unseen.json": {
            "honest_verdict": "success_s5i5_L1_offline_reproduced",
            "new_levels_reproduced": 1,
            "offline_reproduced": True,
            "verifier_is_oracle": True,
            "flagged_adversarial": True,
        },
        "experiment_4423_generic_first_contact_breadth.json": {
            "honest_verdict": "partial: generic_first_contact_g50t_routed_missing_verifier_gap_logged",
            "target_game": "g50t",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "missing_verifier_gaps": [{"gap_id": "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT"}],
            "verifier_is_oracle": False,
        },
        "experiment_4424_deeper_solved_game.json": {
            "honest_verdict": "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap",
            "new_levels_reproduced": 0,
            "offline_reproduced": False,
            "per_mechanic_test_pass_rate": 0.5,
            "residual_failing_mechanic": "sc25_l2_route_search_still_missing_after_hud_cleanup",
            "verifier_is_oracle": True,
        },
        "experiment_4425_config_rule_vocabulary_transfer.json": {
            "honest_verdict": "complete: null_config_rule_vocabulary_transfer_seeded_arm_missing",
            "config_rule_vocabulary_transfers": False,
            "flagged_adversarial": True,
            "verifier_is_oracle": False,
        },
        "experiment_4426_arc_registry_repro_audit.json": {
            "honest_verdict": "complete: registry_repro_audit_total_35_asserted_36_audited",
            "inference_substrate": "offline_arc_registry_repro_audit_cpu_no_llm",
            "all_counted_entries_reproduced": True,
            "registry_claimed_reproducible_total_levels": 35,
            "counted_entries_audited": 18,
            "milestone_409_reproduction_gates": [
                {"experiment": "exp4421", "new_levels_counted": 1, "reproduction_gated": True},
                {"experiment": "exp4423", "new_levels_counted": 0, "reproduction_gated": True},
                {"experiment": "exp4424", "new_levels_counted": 0, "reproduction_gated": True},
            ],
        },
    }
    for filename, payload in payloads.items():
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    assert mod.load_v409_outcomes(tmp_path) == mod.DEFAULT_V409_OUTCOMES


def test_write_outputs_updates_files_idempotently_for_req_report_4429(tmp_path: Path) -> None:
    """REQ-REPORT-4429: writer emits the artifact and studying entry."""

    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    studying = studying_path.read_text(encoding="utf-8")
    assert studying.count("2026-06-19 Exp 4429") == 1
    assert "flagged_for_v410" in studying
    assert "arXiv:2605.05138" in studying
    assert "generic_first_contact_g50t" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4429() -> None:
    """REQ-REPORT-4429: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-19 Exp 4429") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-19 Exp 4429")
    assert studying_refreshed.count("2026-06-19 Exp 4429") == 1
    assert marker_at_end.count("2026-06-19 Exp 4429") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "Executable ARC-AGI-3" in no_heading


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4429(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4429: module and required wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4429_ROOT", str(tmp_path))

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
