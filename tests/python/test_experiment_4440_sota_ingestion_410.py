"""Tests for REQ-REPORT-4440 / SCENARIO-REPORT-4440."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4440_sota_ingestion_410 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4440_sota_ingestion_410.json")
WRAPPER_PATH = Path("results/experiment_4440_sota_ingestion_410.py")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-410-2026-06-19.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods=_valid_methods(),
        flagged_for_v411=mod.DEFAULT_FLAGGED_FOR_V411,
        v410_outcome_conditioning=dict(mod.DEFAULT_V410_OUTCOMES),
        preconditions_checked=dict(mod.DEFAULT_PRECONDITIONS_CHECKED),
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4440_spec_anchor_exists() -> None:
    """REQ-REPORT-4440: OpenSpec declares the .410 SOTA ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4440" in spec
    assert "SCENARIO-REPORT-4440" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert NOTE_PATH.as_posix() in spec
    assert "flagged_for_v411" in spec
    assert "complete: sota_ingestion_410_mapped" in spec
    assert "arXiv:2310.19791" in spec
    assert "Semantic Scholar 429" in spec


def test_build_artifact_has_required_fields_for_req_report_4440() -> None:
    """REQ-REPORT-4440: artifact exposes the requested principle fields."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["methods"] == _valid_methods()
    assert artifact["flagged_for_v411"] == mod.DEFAULT_FLAGGED_FOR_V411
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["research_note_path"] == NOTE_PATH.as_posix()
    assert "SOTA->experiment" in artifact["sota_to_experiment_mapping_note"]


def test_extract_v410_outcomes_reads_source_artifact_fields() -> None:
    """SCENARIO-REPORT-4440: source artifacts determine the .411 hand-off."""

    outcomes = mod.extract_v410_outcomes(
        loo_benchmark={
            "honest_verdict": "complete: generic_loo_solve_count_2_of_7_gate_passed",
            "solve_count": 2,
            "target_count": 7,
            "offline_reproduced": True,
        },
        win_induction={
            "honest_verdict": "success: example_conditioned_g50t_L1_offline_reproduced",
            "target_game": "g50t",
            "offline_reproduced": True,
            "reproduced_levels": 1,
        },
        action_model={
            "honest_verdict": "success: example_conditioning_improved_world_model_accuracy",
            "world_model_accuracy_cold": 0.714286,
            "world_model_accuracy_with_examples": 1.0,
            "reproduced_levels": 0,
        },
        first_contact={
            "honest_verdict": "complete: generic_first_contact_dc22_routed_no_new_level_gap_logged",
            "residual_mechanic_gap_logged": True,
            "offline_reproduced": False,
            "reproduced_levels": 0,
        },
        primitive_consolidation={
            "honest_verdict": "success: tu93_L5_deepened_primitives_consolidated",
            "deepened_game": "tu93",
            "new_levels_reproduced": 1,
            "offline_reproduced": True,
            "primitives_consolidated": [
                {"operator": "glyph_rewrite_matcher"},
                {"operator": "config_rule_grounding"},
                {"operator": "graph_astar_action_cost"},
            ],
        },
    )

    assert outcomes == mod.DEFAULT_V410_OUTCOMES


def test_select_flagged_for_v411_conditions_on_v410_outcomes() -> None:
    """SCENARIO-REPORT-4440: .411 flag follows the .410 branch decision."""

    assert mod.select_flagged_for_v411(mod.DEFAULT_V410_OUTCOMES) == mod.DEFAULT_FLAGGED_FOR_V411
    assert (
        mod.select_flagged_for_v411(
            mod.DEFAULT_V410_OUTCOMES
            | {
                "example_conditioned_win_reproduced": False,
                "example_conditioned_world_model_lift": False,
                "primitives_consolidated": False,
            }
        )
        == mod.EXECUTABLE_WORLD_MODEL_FLAGGED_FOR_V411
    )
    assert (
        mod.select_flagged_for_v411(
            mod.DEFAULT_V410_OUTCOMES
            | {
                "example_conditioned_world_model_lift": True,
                "primitives_consolidated": False,
                "first_contact_gap_open": True,
            }
        )
        == mod.LOOP_OWM_FLAGGED_FOR_V411
    )
    assert (
        mod.select_flagged_for_v411(
            mod.DEFAULT_V410_OUTCOMES
            | {
                "example_conditioned_world_model_lift": False,
                "first_contact_gap_open": True,
                "primitives_consolidated": False,
            }
        )
        == mod.CODEARC_FLAGGED_FOR_V411
    )
    assert (
        mod.select_flagged_for_v411(
            mod.DEFAULT_V410_OUTCOMES
            | {
                "example_conditioned_world_model_lift": False,
                "first_contact_gap_open": False,
                "primitives_consolidated": False,
            }
        )
        == mod.EXECUTABLE_WORLD_MODEL_FLAGGED_FOR_V411
    )


def test_outcome_helpers_cover_fallback_paths_for_req_report_4440() -> None:
    """REQ-REPORT-4440: malformed source summaries never imply strong transfer."""

    per_game = [
        {"solved_without_own_recipe": True},
        {"solved_without_own_recipe": False},
        {"solved_without_own_recipe": True},
    ]

    assert mod._count_solved_without_own_recipe({"per_game": per_game}) == 2
    assert mod._count_solved_without_own_recipe({"per_game": "bad"}) is None
    assert mod._target_count({"per_game": per_game}) == 3
    assert mod._target_count({"per_game": "bad"}) is None


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"inference_substrate": "gpu_training"}, "CPU"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"methods": _valid_methods()[:4]}, "five to eight"),
        (
            _valid_artifact()
            | {
                "methods": [_valid_methods()[0] | {"arxiv_id": "9999.99999"}]
                + _valid_methods()[1:]
            },
            "verified arXiv",
        ),
        (_valid_artifact() | {"methods": [_valid_methods()[0]] + _valid_methods()[:-1]}, "duplicate"),
        (
            _valid_artifact()
            | {"methods": [_valid_methods()[0] | {"pitfalls": ""}] + _valid_methods()[1:]},
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v411": ""}, "flagged_for_v411"),
        (_valid_artifact() | {"flagged_for_v411": "unconditioned"}, "conditioned"),
        (_valid_artifact() | {"random_seed": "4440"}, "random_seed"),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"deep_research_invoked": True}},
            "deep-research",
        ),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"leaderboard_submission": True}},
            "leaderboard",
        ),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"sweep_semscholar_status": ""}},
            "Semantic Scholar",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"cpu_only": False}},
            "CPU",
        ),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"sweep_clusters_urls": []}},
            "cluster URLs",
        ),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"arxiv_api_verified_ids": []}},
            "arXiv ids",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"webfetch_http_200_verified_urls": []}
            },
            "HTTP 200 source URLs",
        ),
        (_valid_artifact() | {"v410_outcome_conditioning": {}}, "v410_outcome_conditioning"),
        (_valid_artifact() | {"sota_to_experiment_mapping_note": "too short"}, "mapping note"),
        (_valid_artifact() | {"research_note_path": "docs/research-notes/wrong.md"}, "research_note_path"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_report_4440(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4440: invalid SOTA artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_method_shape() -> None:
    """REQ-REPORT-4440: artifact and method fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact() | {"url": "https://example.com"}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {"methods": ["not-a-dict"] + _valid_methods()[1:]}
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)


def test_validate_notes_check_citations_and_handoff() -> None:
    """SCENARIO-REPORT-4440: notes keep verified citations and .411 flag."""

    mod.validate_research_note(mod.RESEARCH_NOTE)
    mod.validate_studying_section(mod.STUDYING_SECTION)

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("arXiv:2310.19791", "LILO"))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("No leaderboard submission", "No LB"))
    with pytest.raises(ValueError, match="flagged_for_v411"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("flagged_for_v411", "next flag"))
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("arXiv:2606.12316", "Loop-OWM"))


def test_load_v410_outcomes_reads_repo_relative_sources(tmp_path: Path) -> None:
    """REQ-REPORT-4440: loader reads the five .410 source artifacts."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    source_payloads = {
        "experiment_4432_loo_generic_solve_benchmark.json": {
            "honest_verdict": "complete: generic_loo_solve_count_2_of_7_gate_passed",
            "solve_count": 2,
            "target_count": 7,
            "offline_reproduced": True,
        },
        "experiment_4433_example_conditioned_win_induction.json": {
            "honest_verdict": "success: example_conditioned_g50t_L1_offline_reproduced",
            "target_game": "g50t",
            "offline_reproduced": True,
            "reproduced_levels": 1,
        },
        "experiment_4434_example_conditioned_action_model.json": {
            "honest_verdict": "success: example_conditioning_improved_world_model_accuracy",
            "world_model_accuracy_cold": 0.714286,
            "world_model_accuracy_with_examples": 1.0,
            "reproduced_levels": 0,
        },
        "experiment_4435_generic_first_contact_fixed.json": {
            "honest_verdict": "complete: generic_first_contact_dc22_routed_no_new_level_gap_logged",
            "residual_mechanic_gap_logged": True,
            "offline_reproduced": False,
            "reproduced_levels": 0,
        },
        "experiment_4436_deepen_plus_primitive_consolidation.json": {
            "honest_verdict": "success: tu93_L5_deepened_primitives_consolidated",
            "deepened_game": "tu93",
            "new_levels_reproduced": 1,
            "offline_reproduced": True,
            "primitives_consolidated": [{"operator": "glyph_rewrite_matcher"}],
        },
    }
    for filename, payload in source_payloads.items():
        (results_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    assert mod.load_v410_outcomes(tmp_path) == mod.DEFAULT_V410_OUTCOMES


def test_write_outputs_updates_files_idempotently_for_req_report_4440(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4440: writer emits artifact, research note, and studying entry."""

    artifact_path = tmp_path / ARTIFACT_PATH
    note_path = tmp_path / NOTE_PATH
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        note_path=note_path,
        studying_path=studying_path,
    )
    second_artifact = mod.write_outputs(
        artifact_path=artifact_path,
        note_path=note_path,
        studying_path=studying_path,
    )

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert "SOTA->experiment" in note_path.read_text(encoding="utf-8")
    studying = studying_path.read_text(encoding="utf-8")
    assert studying.count("2026-06-19 Exp 4440") == 1
    assert "flagged_for_v411" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4440() -> None:
    """REQ-REPORT-4440: studying updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-19 Exp 4440") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-19 Exp 4440")
    assert studying_refreshed.count("2026-06-19 Exp 4440") == 1
    assert marker_at_end.count("2026-06-19 Exp 4440") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "LILO" in no_heading


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4440(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4440: module and wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4440_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")
    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()
    assert (tmp_path / NOTE_PATH).exists()

    with pytest.raises(SystemExit) as wrapper_exit:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")
    assert wrapper_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()
