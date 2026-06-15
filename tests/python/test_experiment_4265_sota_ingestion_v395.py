"""Tests for REQ-REPORT-4265 / SCENARIO-REPORT-4265."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4265_sota_ingestion_v395 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-v395-2026-06-15.md")
ARTIFACT_PATH = Path("results/experiment_4265_sota_ingestion_v395.json")
WRAPPER_PATH = Path("results/experiment_4265_sota_ingestion_v395.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v395=mod.DEFAULT_FLAGGED_FOR_V395,
        random_seed=mod.DEFAULT_RANDOM_SEED,
    )


def test_req_report_4265_spec_anchor_exists() -> None:
    """REQ-REPORT-4265: OpenSpec declares the .395 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4265" in spec
    assert "SCENARIO-REPORT-4265" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v395" in spec
    assert "random_seed" in spec
    assert "blocked_arc_game_ids_unrecoverable" in spec
    assert "synthesis_breaks_oracle_ceiling=false" in spec
    assert "blocked_diffusiongemma_gguf_loader_failed" in spec
    for source in mod.VERIFIED_SOURCE_URLS.values():
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4265() -> None:
    """REQ-REPORT-4265: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v395": mod.DEFAULT_FLAGGED_FOR_V395,
        "random_seed": mod.DEFAULT_RANDOM_SEED,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL (no citation = "
                "fabrication per adversarial_verify discipline) + a one-line "
                ".395 experiment mapping."
            ),
            "flagged_for_v395": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the .395 planner, conditioned on the .394 outcomes."
            ),
            "random_seed": (
                "Determinism placeholder for the discovery query set (recorded "
                "for reproducibility of the sweep)."
            ),
        },
    }


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
                        "v394_outcome_conditioning": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2603.05099"}
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
        (_valid_artifact() | {"flagged_for_v395": ""}, "flagged_for_v395"),
        (
            _valid_artifact()
            | {"flagged_for_v395": "diffusiongemma_full_run_guidance_v395"},
            "ARC-TGI",
        ),
        (_valid_artifact() | {"random_seed": "4265"}, "random_seed"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4265(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4265: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4265: artifact fields are exact."""

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


def test_validate_markdown_note_checks_scenario_report_4265_sections() -> None:
    """SCENARIO-REPORT-4265: note maps sources to .394 outcome-conditioned .395 targets."""

    note = """
    ## Fresh-pass provenance
    reliable-channel sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch HTTP 429.
    /deep-research not invoked.
    ## Prior-covered methods not re-ingested
    Compute-as-Teacher GSA Reward-Guided Stitching S3 EDLM 2406.01572 CoDeC
    ARC of Progress ARCTraj Compositional Neuro-Symbolic Reasoning.
    ## .394 fork outcome read
    arc_provenance_blind_win_survives provenance_blind_delta=0.3846153846
    arc_oracle_distinct_win_replicates_multiseed mean_delta=0.4576923077
    blocked_arc_game_ids_unrecoverable
    arc_synthesis_underperforms_selection synthesis_breaks_oracle_ceiling=false
    synthesis_minus_oracle_delta=-0.2826086957
    blocked_diffusiongemma_gguf_loader_failed preflight_go=false
    code_oracle_distinct_replication_corpus_specific code_replication_beats_vote=false.
    ## SOTA -> experiment mapping
    arXiv:2603.05099 arXiv:2606.03305 arXiv:2604.24357 arXiv:2603.12554
    arXiv:2606.12847.
    ARC-TGI Reliability Gap DPRM entropy-guided step selection L-VARC.
    Carnot stack mapping. Failure mode. Experiment mapping. .395.
    ## Flagged for .395
    arc_tgi_family_generator_cross_game_generalization_v395
    random_seed=4265
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4265: note must cite sources and close the loop."""

    with pytest.raises(ValueError, match="Flagged for .395"):
        mod.validate_markdown_note("## Fresh-pass provenance\narXiv:2603.05099\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(mod.NOTE_MARKDOWN.replace("arXiv:2603.05099", "ARC-TGI"))

    with pytest.raises(ValueError, match="not re-ingested"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("Prior-covered methods not re-ingested", "Prior")
        )

    with pytest.raises(ValueError, match="blocked_arc_game_ids_unrecoverable"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("blocked_arc_game_ids_unrecoverable", "cross_game_win")
        )

    with pytest.raises(ValueError, match="synthesis_breaks_oracle_ceiling=false"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace(
                "synthesis_breaks_oracle_ceiling=false",
                "synthesis_breaks_oracle_ceiling=true",
            )
        )

    with pytest.raises(ValueError, match="preflight_go=false"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("preflight_go=false", "preflight_go=true")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4265(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4265: writer emits note, artifact, and studying entry."""

    note_path = tmp_path / "note.md"
    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )
    second_artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    mod.validate_markdown_note(note_path.read_text(encoding="utf-8"))
    saved_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    studying = studying_path.read_text(encoding="utf-8")

    assert saved_artifact == artifact
    assert studying.count("2026-06-15 Exp 4265") == 1
    assert "flagged_for_v395" in studying
    assert "Flagged for .395" in studying
    assert "blocked_arc_game_ids_unrecoverable" in studying
    assert "synthesis_breaks_oracle_ceiling=false" in studying
    assert "preflight_go=false" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4265() -> None:
    """REQ-REPORT-4265: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-15 Exp 4265") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-15 Exp 4265")
    assert studying_refreshed.count("2026-06-15 Exp 4265") == 1
    assert marker_at_end.count("2026-06-15 Exp 4265") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert no_heading.rstrip().endswith(
        "Keep DiffusionGemma as loader repair, not a full-run .395 bet."
    )


def test_main_prints_terminal_verdict_for_req_report_4265(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4265: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *,
        note_path: Path,
        artifact_path: Path,
        studying_path: Path,
    ) -> dict[str, object]:
        calls["note_path"] = note_path
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        return {"honest_verdict": mod.DEFAULT_HONEST_VERDICT}

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert calls["note_path"].as_posix().endswith(NOTE_PATH.as_posix())
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())


def test_module_main_guard_exits_zero_for_req_report_4265(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4265: direct module execution exits after writing outputs."""

    monkeypatch.setenv("CARNOT_EXP4265_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_wrapper_script_runs_module_for_req_report_4265(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4265: required results/ wrapper delegates to the module."""

    monkeypatch.setenv("CARNOT_EXP4265_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()


def test_deliverable_files_validate_against_req_report_4265() -> None:
    """REQ-REPORT-4265: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v395"] == mod.DEFAULT_FLAGGED_FOR_V395
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert "2026-06-15 Exp 4265 - .394 fork SOTA ingestion ingested" in studying
    assert (
        "Flagged for .395: `arc_tgi_family_generator_cross_game_generalization_v395`"
        in studying
    )
