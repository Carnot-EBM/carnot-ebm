"""Tests for REQ-REPORT-4478 / SCENARIO-REPORT-4478."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4478_arc_search_verifier_sota_414 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4478_arc_search_verifier_sota_414.json")
WRAPPER_PATH = Path("results/experiment_4478_arc_search_verifier_sota_414.py")
NOTE_PATH = Path("docs/research-notes/arc-search-verifier-sota-414.md")


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_report_4478_spec_anchor_exists() -> None:
    """REQ-REPORT-4478: OpenSpec declares the ARC search/verifier SOTA note."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4478" in spec
    assert "SCENARIO-REPORT-4478" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert NOTE_PATH.as_posix() in spec
    assert "GAP-ARCH-FEATURES" in spec
    assert "GAP-ARCH-GOAL" in spec
    assert "GAP-ARCH-NO-HIERARCHICAL-SEARCH" in spec
    assert "complete: arc_search_verifier_sota_414_mapped_for_v415" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    assert "arXiv:2512.24156" in spec
    assert "arXiv:2606.12316" in spec
    assert "arXiv:2402.08147" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4478() -> None:
    """REQ-REPORT-4478: artifact exposes the required fields and principles."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["source_ids"] == mod.DEFAULT_SOURCE_IDS
    assert artifact["methods"] == mod.DEFAULT_METHODS
    assert artifact["gap_mapping"] == mod.GAP_MAPPING
    assert artifact["strongest_for_v415"] == mod.DEFAULT_STRONGEST_FOR_V415
    assert artifact["research_note_path"] == NOTE_PATH.as_posix()


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "blocked_network"}, "terminal prefix"),
        (_valid_artifact() | {"inference_substrate": "live_llm_inference"}, "aggregation"),
        (_valid_artifact() | {"offline_reproduced": "false"}, "offline_reproduced"),
        (_valid_artifact() | {"reproduced_levels": "0"}, "reproduced_levels"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"source_ids": mod.DEFAULT_SOURCE_IDS[:4]}, "five to eight"),
        (_valid_artifact() | {"source_ids": [*mod.DEFAULT_SOURCE_IDS[:-1], "9999.99999"]}, "verified arXiv"),
        (_valid_artifact() | {"source_ids": [mod.DEFAULT_SOURCE_IDS[0], *mod.DEFAULT_SOURCE_IDS[:-1]]}, "duplicate"),
        (_valid_artifact() | {"methods": mod.DEFAULT_METHODS[:4]}, "source_ids"),
        (
            _valid_artifact()
            | {
                "methods": [
                    mod.DEFAULT_METHODS[0] | {"mapped_gap": "GAP-OTHER"},
                    *mod.DEFAULT_METHODS[1:],
                ]
            },
            "mapped_gap",
        ),
        (
            _valid_artifact()
            | {
                "gap_mapping": {
                    key: value
                    for key, value in mod.GAP_MAPPING.items()
                    if key != "GAP-ARCH-GOAL"
                }
            },
            "gap_mapping",
        ),
        (_valid_artifact() | {"strongest_for_v415": "unverified"}, "strongest_for_v415"),
        (_valid_artifact() | {"random_seed": "4478"}, "random_seed"),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"deep_research_invoked": True}
            },
            "deep-research",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"live_solve_claim": True}
            },
            "live solve",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
        (_valid_artifact() | {"research_note_path": "docs/research-notes/wrong.md"}, "research_note_path"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_report_4478(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4478: invalid ARC SOTA artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """REQ-REPORT-4478: top-level artifact fields are exact."""

    missing = _valid_artifact()
    missing.pop("source_ids")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"url": "https://example.com"}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)


def test_note_json_round_trips_and_preserves_gap_mapping_for_scenario_4478() -> None:
    """SCENARIO-REPORT-4478: markdown note contains the validated JSON artifact."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "relational/delta" in mod.RESEARCH_NOTE
    assert "goal-vs-dynamics" in mod.RESEARCH_NOTE
    assert "hierarchical/MCTS" in mod.RESEARCH_NOTE
    assert "flagged_for_v415" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("```json", "```text"))
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("arXiv:2606.12316", "Loop-OWM"))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("No live solve", "No run"))


def test_write_outputs_emits_stable_note_and_artifact_for_req_report_4478(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4478: writer emits idempotent JSON and markdown outputs."""

    artifact_path = tmp_path / ARTIFACT_PATH
    note_path = tmp_path / NOTE_PATH

    artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_from_note(note_path.read_text(encoding="utf-8")) == artifact
    assert "GAP-ARCH-FEATURES" in note_path.read_text(encoding="utf-8")


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4478(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4478: module and wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4478_ROOT", str(tmp_path))

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
