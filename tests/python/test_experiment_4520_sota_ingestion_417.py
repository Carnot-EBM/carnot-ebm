"""Tests for REQ-REPORT-4520 / SCENARIO-REPORT-4520."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4520_sota_ingestion_417 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4520_sota_ingestion_417.json")
WRAPPER_PATH = Path("results/experiment_4520_sota_ingestion_417.py")
NOTE_PATH = Path("docs/research-notes/arc-action-efficiency-sota-417.md")


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_report_4520_spec_anchor_exists() -> None:
    """REQ-REPORT-4520: OpenSpec declares the .417 action-efficiency ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4520" in spec
    assert "SCENARIO-REPORT-4520" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert NOTE_PATH.as_posix() in spec
    assert "complete: action_efficiency_sota_417_mapped_for_v418" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    assert "https://huggingface.co/api/models" in spec
    assert "arXiv:2008.09241" in spec
    assert "arXiv:2602.00460" in spec
    assert "arXiv:1511.05952" in spec
    assert "arXiv:2602.05832" in spec
    assert "scripts/research_conductor.py" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4520() -> None:
    """REQ-REPORT-4520: artifact exposes required fields and principles."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert set(artifact["field_principles"]) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["source_ids"] == mod.DEFAULT_SOURCE_IDS
    assert artifact["methods_mapped"] == mod.DEFAULT_METHODS_MAPPED
    assert artifact["citations"] == mod.CITATIONS
    assert artifact["v418_flagged_candidates"] == mod.V418_FLAGGED_CANDIDATES
    assert artifact["research_note_path"] == NOTE_PATH.as_posix()
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert len(artifact["methods_mapped"]) == 5
    assert artifact["preconditions_checked"]["network_precondition_hf_models_exit_0"] is True
    assert artifact["preconditions_checked"]["deep_research_invoked"] is False
    assert artifact["preconditions_checked"]["research_conductor_modified"] is False


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "blocked_network"}, "terminal prefix"),
        (_valid_artifact() | {"inference_substrate": "live_llm_inference"}, "aggregation"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"source_ids": mod.DEFAULT_SOURCE_IDS[:4]}, "five to eight"),
        (
            _valid_artifact()
            | {"source_ids": [*mod.DEFAULT_SOURCE_IDS[:-1], "9999.99999"]},
            "verified arXiv",
        ),
        (
            _valid_artifact()
            | {"source_ids": [mod.DEFAULT_SOURCE_IDS[0], *mod.DEFAULT_SOURCE_IDS[:-1]]},
            "duplicate",
        ),
        (_valid_artifact() | {"methods_mapped": mod.DEFAULT_METHODS_MAPPED[:2]}, "three to five"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0]
                    | {"takes_over_current_stack": "unmapped"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "current stack",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"source_ids": ["9999.99999"]},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "verified citations",
        ),
        (
            _valid_artifact()
            | {"citations": {k: v for k, v in mod.CITATIONS.items() if k != "2602.05832"}},
            "citations",
        ),
        (
            _valid_artifact()
            | {"v418_flagged_candidates": ["affordance-pruned frame-change"]},
            "v418_flagged_candidates",
        ),
        (_valid_artifact() | {"random_seed": "4520"}, "random_seed"),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"network_precondition_hf_models_exit_0": False}
            },
            "network",
        ),
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
                | {"research_conductor_modified": True}
            },
            "research_conductor",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
        (_valid_artifact() | {"research_note_path": "docs/research-notes/wrong.md"}, "research_note_path"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_report_4520(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4520: invalid action-efficiency artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """REQ-REPORT-4520: top-level artifact fields are exact."""

    missing = _valid_artifact()
    missing.pop("citations")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"offline_reproduced": False}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)


def test_note_json_round_trips_and_preserves_mapping_for_scenario_4520() -> None:
    """SCENARIO-REPORT-4520: markdown note contains the validated JSON artifact."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "action-effect/clickability" in mod.RESEARCH_NOTE
    assert "offline-search + lazy value head + frame-change predictor" in mod.RESEARCH_NOTE
    assert "bottom line for the .418 roadmap" in mod.RESEARCH_NOTE
    assert "flagged_for_v418" in mod.RESEARCH_NOTE
    assert "HTTP 429" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("```json", "```text"))
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("arXiv:2602.05832", "UI-Mem"))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("No training", "No run"))


def test_write_outputs_emits_stable_files_for_req_report_4520(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4520: writer emits idempotent JSON, note, and queue update."""

    artifact_path = tmp_path / ARTIFACT_PATH
    note_path = tmp_path / NOTE_PATH
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\nExisting queue.\n", encoding="utf-8")

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

    studying = studying_path.read_text(encoding="utf-8")
    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_from_note(note_path.read_text(encoding="utf-8")) == artifact
    assert studying.count(mod.STUDYING_SECTION_START) == 1
    assert "INGESTED into `docs/research-notes/arc-action-efficiency-sota-417.md`" in studying
    assert "flagged_for_v418" in studying


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4520(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4520: module and wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4520_ROOT", str(tmp_path))
    (tmp_path / "research-studying.md").write_text("# Research Studying\n", encoding="utf-8")

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
