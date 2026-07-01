"""Tests for Exp 5070 V466 SOTA ingestion backfill.

Spec refs: REQ-REPORT-5070, SCENARIO-REPORT-5070,
SCENARIO-REPORT-5070-MISSING-REFERENCE.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_5070_sota_ingestion_backfill_v466 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _references_text() -> str:
    return REFERENCES_PATH.read_text(encoding="utf-8")


def _artifact() -> dict[str, object]:
    return mod.build_artifact(reference_text=_references_text())


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5070")
    end = spec.index("### REQ-REPORT-4873", start)
    return spec[start:end]


def test_req_report_5070_spec_declares_backfill_contract() -> None:
    """REQ-REPORT-5070: OpenSpec anchors the V466 backfill contract."""

    section = _spec_section()

    assert "REQ-REPORT-5070" in section
    assert "SCENARIO-REPORT-5070" in section
    assert "SCENARIO-REPORT-5070-MISSING-REFERENCE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.REFERENCES_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "scripts/research_conductor.py" in section
    assert "OpenReview/GitHub" in section
    assert "Semantic Scholar EBT/ARM" in section
    for requirement in mod.REQUIRED_REFERENCE_CHECKS:
        assert requirement["required_token"] in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_report_5070_v466_section_contains_required_sources() -> None:
    """SCENARIO-REPORT-5070: V466 source set has required URLs and hooks."""

    section = mod.extract_v466_section(_references_text())
    verification = mod.verify_v466_references(section)

    assert verification["references_section_found"] is True
    assert verification["missing"] == []
    assert [row["required_token"] for row in verification["present"]] == [
        row["required_token"] for row in mod.REQUIRED_REFERENCE_CHECKS
    ]
    assert "OpenReview" in section
    assert "Semantic Scholar API citation checks" in section
    assert "HTTP 429" in section
    assert "Carnot hook" in section


def test_req_report_5070_artifact_fields_and_principles_are_exact() -> None:
    """REQ-REPORT-5070: artifact emits required principle-annotated fields."""

    artifact = _artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["duration_s"] == mod.DURATION_S
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["references_section_found"] is True
    assert artifact["references_added_count"] == 0
    assert artifact["semantic_scholar_status"] == mod.SEMANTIC_SCHOLAR_STATUS
    assert artifact["flagged_adversarial"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == mod.SPEC_REFS

    channels = {row["channel"] for row in artifact["sources_checked"]}
    assert mod.REQUIRED_CHANNELS.issubset(channels)
    assert len(artifact["sources_checked"]) >= len(mod.REQUIRED_REFERENCE_CHECKS)
    assert all(row["url"].startswith("https://") for row in artifact["sources_checked"])
    assert any(row["channel"] == "OpenReview" for row in artifact["sources_checked"])
    assert any(row["channel"] == "GitHub" for row in artifact["sources_checked"])

    hooks = {hook["hook_id"] for hook in artifact["planning_hooks"]}
    assert {
        "exp5073_uprm_selector",
        "exp5074_vpr_diagnostic",
        "exp5075_dccd_guided_scale",
        "exp5077_guarded_fr11_memory",
        "exp5080_kan_pwa_milp_bridge",
    }.issubset(hooks)

    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("needle", "message"),
    [
        ("- **Source:** arXiv:2605.10158", "2605.10158"),
        ("https://github.com/avinashreddydev/dccd", "github.com/avinashreddydev/dccd"),
        ("Semantic Scholar API citation checks", "Semantic Scholar"),
        ("Carnot hook:** `.466`", "Carnot hook"),
    ],
)
def test_scenario_report_5070_missing_reference_fails_closed(needle: str, message: str) -> None:
    """SCENARIO-REPORT-5070-MISSING-REFERENCE: missing evidence is rejected."""

    text = _references_text().replace(needle, "REMOVED", 1)

    with pytest.raises(ValueError, match=message):
        mod.build_artifact(reference_text=text)


def test_write_outputs_is_stable_and_does_not_edit_references(tmp_path: Path) -> None:
    """REQ-REPORT-5070: writer emits stable JSON without rewriting references."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    original_references = _references_text()
    references_path.write_text(original_references, encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, references_path=references_path)
    second = mod.write_outputs(artifact_path=artifact_path, references_path=references_path)

    assert second == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert references_path.read_text(encoding="utf-8") == original_references
    mod.validate_artifact(artifact)


def test_main_writes_default_artifact_for_req_report_5070(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-5070: direct module execution writes the backfill artifact."""

    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text(_references_text(), encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP5070_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)


def test_deliverable_file_validates_for_req_report_5070() -> None:
    """REQ-REPORT-5070: committed Exp 5070 deliverable satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["references_added_count"] == 0
