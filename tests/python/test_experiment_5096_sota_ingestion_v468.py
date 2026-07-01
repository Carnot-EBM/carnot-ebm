"""Tests for Exp 5096 V468 SOTA ingestion audit.

Spec refs: REQ-REPORT-5096, SCENARIO-REPORT-5096,
SCENARIO-REPORT-5096-MISSING-REFERENCE.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_5096_sota_ingestion_v468 as mod


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
    start = spec.index("### REQ-REPORT-5096")
    end = spec.index("### REQ-REPORT-4873", start)
    return spec[start:end]


def test_req_report_5096_spec_declares_v468_audit_contract() -> None:
    """REQ-REPORT-5096: OpenSpec anchors the V468 audit contract."""

    section = _spec_section()

    assert "REQ-REPORT-5096" in section
    assert "SCENARIO-REPORT-5096" in section
    assert "SCENARIO-REPORT-5096-MISSING-REFERENCE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.REFERENCES_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "scripts/research_conductor.py" in section
    assert "Hugging Face Papers/GitHub" in section
    assert "Semantic Scholar EBT/ARM citation-lineage status" in section
    for requirement in mod.REQUIRED_REFERENCE_CHECKS:
        for token in requirement["spec_tokens"]:
            assert token in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_report_5096_v468_section_contains_required_sources() -> None:
    """SCENARIO-REPORT-5096: V468 source set has URLs, hooks, and actions."""

    section = mod.extract_v468_section(_references_text())
    verification = mod.verify_v468_references(section)

    assert verification["references_section_found"] is True
    assert verification["missing"] == []
    assert [row["source_id"] for row in verification["present"]] == [
        row["source_id"] for row in mod.REQUIRED_REFERENCE_CHECKS
    ]
    assert "Search coverage: arXiv, OpenReview, Hugging Face Papers, GitHub" in section
    assert "Extropic, Logical Intelligence, and" in section
    assert "Scholar API returned HTTP 429" in section
    assert "citation-lineage notes below come from public" in section
    assert section.count("- **Tracks:**") >= len(mod.REQUIRED_REFERENCE_CHECKS)
    assert section.count("- **Carnot hook:**") >= len(mod.REQUIRED_REFERENCE_CHECKS)
    assert section.count("- **Actionability:**") >= len(mod.REQUIRED_REFERENCE_CHECKS)


def test_scenario_report_5096_missing_hook_and_action_counts_are_reported() -> None:
    """SCENARIO-REPORT-5096-MISSING-REFERENCE: hook/action count gaps are rejected."""

    section = (
        mod.extract_v468_section(_references_text())
        .replace("- **Carnot hook:** BEAVER computes", "REMOVED", 1)
        .replace("- **Actionability:** Build", "REMOVED", 1)
    )

    verification = mod.verify_v468_references(section)
    missing_ids = {row["source_id"] for row in verification["missing"]}

    assert "per_source_carnot_hooks" in missing_ids
    assert "per_source_actionability" in missing_ids


def test_req_report_5096_artifact_fields_mapping_and_principles_are_exact() -> None:
    """REQ-REPORT-5096: artifact emits required principle-annotated fields."""

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
    assert mapping_by_source["beaver_prefix_bounds"]["task_id"] == "exp5099"
    assert mapping_by_source["graph_evidence_grounding"]["task_id"] == "exp5101"
    assert mapping_by_source["constrainprompt_code_assurance"]["task_id"] == "exp5100"
    assert mapping_by_source["hubo_pspin_planck"]["task_id"] == "exp5102"
    assert mapping_by_source["taco_adaptive_csp"]["task_id"] == "exp5103"
    assert mapping_by_source["cfg_constrained_diffusion"]["task_id"] == "exp5104"
    assert mapping_by_source["severa_self_evolving_agents"]["task_id"] == "exp5105"
    assert mapping_by_source["neuromorphic_csp_hardware"]["task_id"] == "exp5106"
    assert mapping_by_source["halt_logprob_timeseries"]["task_id"] == "background_only"
    assert mapping_by_source["ebt_arm_citation_lineage"]["task_id"] == "background_only"

    background_ids = {row["source_id"] for row in artifact["background_only_sources"]}
    assert {"halt_logprob_timeseries", "ebt_arm_citation_lineage"}.issubset(
        background_ids
    )

    hooks = {hook["hook_id"] for hook in artifact["planning_hooks"]}
    assert {
        "exp5099_beaver_prefix_bounds",
        "exp5100_prompt_code_assurance",
        "exp5101_graph_evidence_energy",
        "exp5102_hubo_pspin_direct_energy",
        "exp5103_taco_adaptive_csp",
        "exp5104_constrained_decoding_semantic_audit",
        "exp5105_severa_fr11_memory",
        "exp5106_hardware_partition_telemetry",
    }.issubset(hooks)

    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("needle", "message"),
    [
        ("- **Source:** arXiv:2512.05439", "beaver_prefix_bounds"),
        ("https://github.com/uiuc-focal-lab/Beaver", "beaver_prefix_bounds"),
        ("- **Source:** arXiv:2606.30247", "graph_evidence_grounding"),
        (
            "- **Source:** OpenReview - https://openreview.net/forum?id=O3Kg4dLdpg",
            "constrainprompt",
        ),
        ("Scholar API returned HTTP 429 during the planner scan", "Semantic Scholar"),
    ],
)
def test_scenario_report_5096_missing_reference_fails_closed(
    needle: str, message: str
) -> None:
    """SCENARIO-REPORT-5096-MISSING-REFERENCE: missing evidence is rejected."""

    text = _references_text().replace(needle, "REMOVED", 1)

    with pytest.raises(ValueError, match=message):
        mod.build_artifact(reference_text=text)


def test_write_outputs_is_stable_and_does_not_edit_references(tmp_path: Path) -> None:
    """REQ-REPORT-5096: writer emits stable JSON without rewriting references."""

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


def test_main_writes_default_artifact_for_req_report_5096(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-5096: direct module execution writes the V468 artifact."""

    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text(_references_text(), encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP5096_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)


def test_deliverable_file_validates_for_req_report_5096() -> None:
    """REQ-REPORT-5096: committed Exp 5096 deliverable satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["references_added_count"] == 0
