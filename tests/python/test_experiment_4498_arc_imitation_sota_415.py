"""Tests for REQ-REPORT-4498 / SCENARIO-REPORT-4498."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4498_arc_imitation_sota_415 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4498_arc_imitation_sota_415.json")
WRAPPER_PATH = Path("results/experiment_4498_arc_imitation_sota_415.py")
NOTE_PATH = Path("docs/research-notes/arc-imitation-sota-415.md")


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_report_4498_spec_anchor_exists() -> None:
    """REQ-REPORT-4498: OpenSpec declares the ARC imitation SOTA note."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4498" in spec
    assert "SCENARIO-REPORT-4498" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert NOTE_PATH.as_posix() in spec
    assert "14,672" in spec
    assert "5x priority" in spec
    assert "GAP-ARCH-FRAME-CHANGE-PREDICTOR" in spec
    assert "GAP-ARCH-VALUE-ENERGY-HEADS" in spec
    assert "GAP-ARCH-EXPERT-INJECTION-REPLAY" in spec
    assert "complete: arc_imitation_sota_415_mapped_for_v416" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    assert "arXiv:1704.03732" in spec
    assert "arXiv:1511.05952" in spec
    assert "arXiv:2206.11795" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4498() -> None:
    """REQ-REPORT-4498: artifact exposes required fields and principles."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert set(artifact["field_principles"]) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["source_ids"] == mod.DEFAULT_SOURCE_IDS
    assert artifact["methods"] == mod.DEFAULT_METHODS
    assert artifact["human_corpus"] == mod.HUMAN_CORPUS
    assert artifact["leaderboard_dqn_mapping"] == mod.LEADERBOARD_DQN_MAPPING
    assert artifact["arc_mapping"] == mod.ARC_MAPPING
    assert artifact["strongest_for_v416"] == mod.DEFAULT_STRONGEST_FOR_V416
    assert artifact["research_note_path"] == NOTE_PATH.as_posix()
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert artifact["human_corpus"]["example_count"] == 14672
    assert artifact["human_corpus"]["frame_changing_actions"] == 14243
    assert artifact["leaderboard_dqn_mapping"]["expert_priority_multiplier"] == 5


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
        (_valid_artifact() | {"methods": mod.DEFAULT_METHODS[:4]}, "source_ids"),
        (
            _valid_artifact()
            | {
                "methods": [
                    mod.DEFAULT_METHODS[0] | {"mapped_application": "GAP-OTHER"},
                    *mod.DEFAULT_METHODS[1:],
                ]
            },
            "mapped_application",
        ),
        (
            _valid_artifact()
            | {
                "arc_mapping": {
                    key: value
                    for key, value in mod.ARC_MAPPING.items()
                    if key != "GAP-ARCH-VALUE-ENERGY-HEADS"
                }
            },
            "arc_mapping",
        ),
        (
            _valid_artifact()
            | {"human_corpus": mod.HUMAN_CORPUS | {"example_count": 342}},
            "human_corpus",
        ),
        (
            _valid_artifact()
            | {
                "leaderboard_dqn_mapping": mod.LEADERBOARD_DQN_MAPPING
                | {"expert_priority_multiplier": 1}
            },
            "expert injection",
        ),
        (_valid_artifact() | {"strongest_for_v416": "unverified"}, "strongest_for_v416"),
        (_valid_artifact() | {"random_seed": "4498"}, "random_seed"),
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
                | {"training_launched": True}
            },
            "training",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
        (
            _valid_artifact()
            | {"research_note_path": "docs/research-notes/wrong.md"},
            "research_note_path",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_report_4498(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4498: invalid ARC imitation artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """REQ-REPORT-4498: top-level artifact fields are exact."""

    missing = _valid_artifact()
    missing.pop("human_corpus")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"offline_reproduced": False}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)


def test_note_json_round_trips_and_preserves_mapping_for_scenario_4498() -> None:
    """SCENARIO-REPORT-4498: markdown note contains the validated JSON artifact."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "behavior cloning" in mod.RESEARCH_NOTE
    assert "offline RL" in mod.RESEARCH_NOTE
    assert "prioritized replay" in mod.RESEARCH_NOTE
    assert "expert-injection" in mod.RESEARCH_NOTE
    assert "14,672-example" in mod.RESEARCH_NOTE
    assert "flagged_for_v416" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("```json", "```text"))
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("arXiv:1704.03732", "DQfD"))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("No training", "No run"))


def test_write_outputs_emits_stable_note_and_artifact_for_req_report_4498(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4498: writer emits idempotent JSON and markdown outputs."""

    artifact_path = tmp_path / ARTIFACT_PATH
    note_path = tmp_path / NOTE_PATH

    artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_from_note(note_path.read_text(encoding="utf-8")) == artifact
    assert "GAP-ARCH-FRAME-CHANGE-PREDICTOR" in note_path.read_text(encoding="utf-8")


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4498(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4498: module and wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4498_ROOT", str(tmp_path))

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
