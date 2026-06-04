"""Tests for Exp 3794 external research refresh.

Spec refs: REQ-REPORT-3794, SCENARIO-REPORT-3794.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import external_research_refresh_3794 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

PRIOR_REFERENCES = """# Research References

## .345 additions - existing abstention section

- Existing .345 content stays untouched.

## .346 additions - existing verifier robustness section

- Existing .346 content stays untouched.
"""


def _seed_references(root: Path, text: str = PRIOR_REFERENCES) -> str:
    path = root / mod.RESEARCH_REFERENCES_REL_PATH
    path.write_text(text, encoding="utf-8")
    return text


def test_req_report_3794_spec_anchor_exists() -> None:
    """REQ-REPORT-3794: OpenSpec declares the refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3794" in spec
    assert "SCENARIO-REPORT-3794" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.RESEARCH_REFERENCES_REL_PATH.as_posix() in spec
    for arxiv_id in mod.REFERENCE_IDS:
        assert arxiv_id in spec
    assert "arXiv:2604.15149" in spec


def test_scenario_report_3794_appends_references_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3794: the .347 section is append-only and auditable."""

    before = _seed_references(tmp_path)

    out_path = mod.run(tmp_path, started_s=100.0, now_s=100.25)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    after = (tmp_path / mod.RESEARCH_REFERENCES_REL_PATH).read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert after.startswith(before)
    assert after.count("\n## .347 additions") == 1
    assert "reference_deep_think_post_bounded_2026_06" in after
    assert "project_energy_selection_thesis_bounded" in after
    assert "project_thesis_a_ebt_seeded" in after
    assert "Numbers are source-reported" in after
    for arxiv_id in mod.REFERENCE_IDS:
        assert arxiv_id in after
    assert "arXiv:2604.15149" not in mod.REFERENCE_IDS
    for track in (
        "EDLM next-thesis preflight",
        "selective-prediction / abstention surface",
        "Tier-3 predictive verification",
        "gaming-resistance",
    ):
        assert track in after

    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["references_added"] == list(mod.REFERENCE_IDS)
    assert artifact["n_references_added"] == len(mod.REFERENCE_IDS)
    assert artifact["section_appended_not_replaced"] is True
    assert artifact["numbers_are_as_reported"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == 0.25
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["section_sha256"] == mod.sha256_text(mod.render_section())
    assert artifact["adversarial_verify_clean"] is True
    assert {row["arxiv_id"] for row in artifact["references"]} == set(mod.REFERENCE_IDS)
    assert all(row["arxiv_abs_resolved"] is True for row in artifact["references"])
    assert all(row["numbers_are_as_reported"] is True for row in artifact["references"])
    assert all(row["source_kind"] == "arxiv_abs" for row in artifact["references"])

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded


def test_req_report_3794_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3794: a rerun does not duplicate or replace the section."""

    before = _seed_references(tmp_path)
    path = tmp_path / mod.RESEARCH_REFERENCES_REL_PATH

    first = mod.append_section_if_missing(path)
    second = mod.append_section_if_missing(path)
    after = path.read_text(encoding="utf-8")

    assert first == "appended"
    assert second == "already_present"
    assert after.startswith(before)
    assert after.count(mod.SECTION_HEADER) == 1


def test_req_report_3794_helper_fallbacks_are_explicit() -> None:
    """REQ-REPORT-3794: verification-report helper fallbacks are deterministic."""

    assert mod.report_is_clean(None) is True
    assert mod.severity_rank("unknown") == -1
    assert mod.compact_verify_report({"flags": [{"severity": "warn"}]}) == {
        "max_severity": 1,
        "flags": [{"severity": "warn"}],
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact.pop("references_added"), "missing required"),
        (lambda artifact: artifact.update(honest_verdict="complete: wrong"), "honest_verdict"),
        (lambda artifact: artifact.update(n_references_added=3), "n_references_added"),
        (
            lambda artifact: artifact.update(section_appended_not_replaced=False),
            "section_appended_not_replaced",
        ),
        (
            lambda artifact: artifact.update(numbers_are_as_reported=False),
            "numbers_are_as_reported",
        ),
        (
            lambda artifact: artifact["references"].append(
                {
                    "arxiv_id": "arXiv:9999.99999",
                    "arxiv_abs_resolved": True,
                    "numbers_are_as_reported": True,
                }
            ),
            "references",
        ),
        (lambda artifact: artifact.update(reproducibility_checksum="bad"), "checksum"),
    ],
)
def test_req_report_3794_validation_rejects_schema_violations(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3794: validation enforces the required artifact fields."""

    _seed_references(tmp_path)
    artifact = mod.build_artifact(
        tmp_path,
        append_action="appended",
        before_text=PRIOR_REFERENCES,
        after_text=PRIOR_REFERENCES + "\n" + mod.render_section(),
        started_s=1.0,
        now_s=1.25,
    )
    mutate(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_scenario_report_3794_script_entrypoint_exists() -> None:
    """SCENARIO-REPORT-3794: the requested script entrypoint exists."""

    assert Path("scripts/experiment_3794_external_research_refresh.py").exists()
