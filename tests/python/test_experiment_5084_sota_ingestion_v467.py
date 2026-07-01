"""Tests for Exp 5084 V467 SOTA ingestion audit.

Spec refs: REQ-REPORT-5084, SCENARIO-REPORT-5084,
SCENARIO-REPORT-5084-MISSING-REFERENCE.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_5084_sota_ingestion_v467 as mod


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
    start = spec.index("### REQ-REPORT-5084")
    end = spec.index("### REQ-REPORT-4873", start)
    return spec[start:end]


def test_req_report_5084_spec_declares_v467_audit_contract() -> None:
    """REQ-REPORT-5084: OpenSpec anchors the V467 audit contract."""

    section = _spec_section()

    assert "REQ-REPORT-5084" in section
    assert "SCENARIO-REPORT-5084" in section
    assert "SCENARIO-REPORT-5084-MISSING-REFERENCE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.REFERENCES_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "scripts/research_conductor.py" in section
    assert "Hugging Face Papers/GitHub" in section
    assert "Semantic Scholar EBT/ARM citation-lineage metadata" in section
    for requirement in mod.REQUIRED_REFERENCE_CHECKS:
        for token in requirement["spec_tokens"]:
            assert token in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_report_5084_v467_section_contains_required_sources() -> None:
    """SCENARIO-REPORT-5084: V467 source set has URLs, hooks, and actions."""

    section = mod.extract_v467_section(_references_text())
    verification = mod.verify_v467_references(section)

    assert verification["references_section_found"] is True
    assert verification["missing"] == []
    assert [row["source_id"] for row in verification["present"]] == [
        row["source_id"] for row in mod.REQUIRED_REFERENCE_CHECKS
    ]
    assert "Search coverage: arXiv, OpenReview, Hugging Face Papers, GitHub" in section
    assert "Semantic Scholar returned live metadata" in section
    assert "EBT had 26 listed citations" in section
    assert "ARM-EBM had 7 listed citations" in section
    assert section.count("- **Tracks:**") >= len(mod.REQUIRED_REFERENCE_CHECKS) - 1
    assert section.count("- **Carnot hook:**") >= len(mod.REQUIRED_REFERENCE_CHECKS) - 1
    assert section.count("- **Actionability:**") >= len(mod.REQUIRED_REFERENCE_CHECKS) - 1


def test_req_report_5084_artifact_fields_mapping_and_principles_are_exact() -> None:
    """REQ-REPORT-5084: artifact emits required principle-annotated fields."""

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

    channels = {
        channel for row in artifact["sources_checked"] for channel in row["channels"]
    }
    assert mod.REQUIRED_CHANNELS.issubset(channels)
    assert len(artifact["sources_checked"]) >= len(mod.REQUIRED_REFERENCE_CHECKS)
    assert all(row["urls"] for row in artifact["sources_checked"])
    assert any("OpenReview" in row["channels"] for row in artifact["sources_checked"])
    assert any("GitHub" in row["channels"] for row in artifact["sources_checked"])

    mapping_ids = [row["source_id"] for row in artifact["task_mapping"]]
    assert mapping_ids == [row["source_id"] for row in mod.REQUIRED_REFERENCE_CHECKS]
    mapping_by_source = {row["source_id"]: row for row in artifact["task_mapping"]}
    assert mapping_by_source["pbit_guided_cdcl"]["task_id"] == "exp5089"
    assert mapping_by_source["static_csr_constrained_decoding"]["task_id"] == "exp5090"
    assert mapping_by_source["temporal_consistency_prm"]["task_id"] == "exp5088"
    assert mapping_by_source["memorybench_procedural_memory"]["task_id"] == "exp5092"
    assert mapping_by_source["fixed_point_reasoners_loopus"]["task_id"] == "background_only"
    assert mapping_by_source["extropic_xtr0_tsu"]["task_id"] == "background_only"

    hooks = {hook["hook_id"] for hook in artifact["planning_hooks"]}
    assert {
        "exp5088_temporal_consistency_fallback",
        "exp5089_pbit_cdcl_bridge",
        "exp5090_static_csr_masks",
        "exp5091_kan_mip_exact_verifier",
        "exp5092_governed_fr11_memory",
        "exp5093_hardware_continuity_telemetry",
    }.issubset(hooks)

    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("needle", "message"),
    [
        ("- **Source:** arXiv:2606.25313", "pbit_million_scale_hardware"),
        ("- **Code:** https://github.com/youtube/static-constraint-decoding", "static_csr"),
        ("https://openreview.net/forum?id=sM5QDzIg3j", "temporal_consistency"),
        ("EBT had 26 listed citations", "Semantic Scholar"),
        ("- **Actionability:** SmartSnap", "smartsnap"),
    ],
)
def test_scenario_report_5084_missing_reference_fails_closed(
    needle: str, message: str
) -> None:
    """SCENARIO-REPORT-5084-MISSING-REFERENCE: missing evidence is rejected."""

    text = _references_text().replace(needle, "REMOVED", 1)

    with pytest.raises(ValueError, match=message):
        mod.build_artifact(reference_text=text)


def test_write_outputs_is_stable_and_does_not_edit_references(tmp_path: Path) -> None:
    """REQ-REPORT-5084: writer emits stable JSON without rewriting references."""

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


def test_main_writes_default_artifact_for_req_report_5084(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-5084: direct module execution writes the V467 artifact."""

    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text(_references_text(), encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP5084_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)


def test_deliverable_file_validates_for_req_report_5084() -> None:
    """REQ-REPORT-5084: committed Exp 5084 deliverable satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["references_added_count"] == 0
