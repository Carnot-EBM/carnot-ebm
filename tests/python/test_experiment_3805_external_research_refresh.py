"""Tests for Exp 3805 external research refresh.

Spec refs: REQ-REPORT-3805, SCENARIO-REPORT-3805.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import external_research_refresh_3805 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

PRIOR_REFERENCES = """# Research References

## .347 additions - existing prior section

- Existing .347 content stays untouched.

## .348 additions - early-rejection predictive verification, outcome-guided process rewards, EDLM impl confirmation (2026-06-04)

Added by the `.348 planning sweep. Append-only into the converged record: `paper_ready` stays TRUE (G1-G4), FoVer 0.9131 stays frozen, both energy routes stay bounded per `[[project_energy_selection_thesis_bounded]]` / `[[project_thesis_a_ebt_seeded]]`, the verifier-moat thread stays closed per `[[reference_deep_think_post_bounded_2026_06]]`, and EDLM stays an operator-seeded preflight route (the `.347 exp3793 preflight returned GO; the loop does NOT self-seed). `.348 is a LEAN, NON-CHURN maintenance milestone: it ADVANCES the headline (G4 restoration of the demoted product code-repair number), HARDENS the banked verifier product, TUNES the over-firing anomaly classifier, and APPLIES the `.347 Tier-3 predictor as a real fast-path gate.

Numbers are source-reported by the papers. They are not Carnot measurements and must be independently re-derived before entering any forward-facing claim.

- **arXiv:2508.01969 - "Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling":** Track: FR-11 v20 Tier-3-as-fast-path. Existing anchor stays.
- **arXiv:2604.02341 - "LLM Reasoning with Process Rewards for Outcome-Guided Steps" (submitted 2026-02-08):** Track: Tier-3 predictive verification + the banked verifier product. Existing anchor stays.
- **EDLM reference implementation confirmed (GitHub: MinkaiXu/Energy-Diffusion-LLM; arXiv:2410.21357, ICLR 2025):** Track: EDLM operator-seed. Existing anchor stays.
"""


def _seed_references(root: Path, text: str = PRIOR_REFERENCES) -> str:
    path = root / mod.RESEARCH_REFERENCES_REL_PATH
    path.write_text(text, encoding="utf-8")
    return text


def test_req_report_3805_spec_anchor_exists() -> None:
    """REQ-REPORT-3805: OpenSpec declares the refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3805" in spec
    assert "SCENARIO-REPORT-3805" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.RESEARCH_REFERENCES_REL_PATH.as_posix() in spec
    for arxiv_id in mod.REFERENCE_IDS:
        assert arxiv_id in spec
    assert "arXiv:2508.01969" in spec
    assert "arXiv:2604.15149" in spec
    assert "arXiv:2602.01750" in spec


def test_scenario_report_3805_appends_into_existing_348_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3805: the .348 section is confirmed and extended."""

    before = _seed_references(tmp_path)

    out_path = mod.run(tmp_path, started_s=100.0, now_s=100.25)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    after = (tmp_path / mod.RESEARCH_REFERENCES_REL_PATH).read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert after.startswith(before)
    assert after.count(mod.SECTION_HEADER) == 1
    assert after.count("arXiv:2508.01969") == 1
    assert "arXiv:2604.15149" not in mod.REFERENCE_IDS
    assert "arXiv:2602.01750" not in mod.REFERENCE_IDS
    assert "Numbers are source-reported by the papers" in after
    assert "MinkaiXu/Energy-Diffusion-LLM" in after
    for arxiv_id in mod.REFERENCE_IDS:
        assert arxiv_id in after
    for track in (
        "FR-11 v20 Tier-3-as-fast-path",
        "verifier robustness / reward-gaming mitigation",
        "selective-prediction / abstention surface",
        "EDLM operator-seed",
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
    assert artifact["section_confirmed_intact"] is True
    assert artifact["section_sha256"] == mod.sha256_text(mod.extract_348_section(after))
    assert artifact["adversarial_verify_clean"] is True
    assert {row["arxiv_id"] for row in artifact["references"]} == set(mod.REFERENCE_IDS)
    assert all(row["arxiv_abs_resolved"] is True for row in artifact["references"])
    assert all(row["numbers_are_as_reported"] is True for row in artifact["references"])
    assert all(row["source_kind"] == "arxiv_abs" for row in artifact["references"])

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded


def test_req_report_3805_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3805: a rerun does not duplicate the .348 filings."""

    before = _seed_references(tmp_path)
    path = tmp_path / mod.RESEARCH_REFERENCES_REL_PATH

    first = mod.append_references_if_missing(path)
    second = mod.append_references_if_missing(path)
    after = path.read_text(encoding="utf-8")

    assert first == "appended"
    assert second == "already_present"
    assert after.startswith(before)
    assert after.count(mod.SECTION_HEADER) == 1
    for arxiv_id in mod.REFERENCE_IDS:
        assert after.count(arxiv_id) == 1


def test_req_report_3805_section_integrity_is_checked(tmp_path: Path) -> None:
    """REQ-REPORT-3805: missing .348 anchors fail before appending."""

    path = tmp_path / mod.RESEARCH_REFERENCES_REL_PATH
    path.write_text("# Research References\n\n## .348 additions - incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="intact"):
        mod.append_references_if_missing(path)

    with pytest.raises(ValueError, match="section"):
        mod.extract_348_section("# Research References\n")

    section_plus_next = PRIOR_REFERENCES + "\n## .349 additions - future\n\n- untouched\n"
    extracted = mod.extract_348_section(section_plus_next)
    assert "## .349 additions" not in extracted

    with pytest.raises(ValueError, match="missing"):
        mod.confirm_348_section_intact(PRIOR_REFERENCES.replace("arXiv:2604.02341", ""))


def test_req_report_3805_helper_fallbacks_are_explicit() -> None:
    """REQ-REPORT-3805: verification-report helper fallbacks are deterministic."""

    assert mod.report_is_clean(None) is True
    assert mod.report_is_clean({"flags": [{"severity": "critical"}]}) is False
    assert mod.severity_rank("unknown") == -1
    assert mod.elapsed_seconds(2.0, 1.0) == 0.0001
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
            lambda artifact: artifact.update(section_confirmed_intact=False),
            "section_confirmed_intact",
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
def test_req_report_3805_validation_rejects_schema_violations(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3805: validation enforces the required artifact fields."""

    before = _seed_references(tmp_path)
    after = before + "\n" + mod.render_reference_bullets()
    artifact = mod.build_artifact(
        tmp_path,
        append_action="appended",
        before_text=before,
        after_text=after,
        started_s=1.0,
        now_s=1.25,
    )
    mutate(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_scenario_report_3805_script_entrypoint_exists() -> None:
    """SCENARIO-REPORT-3805: the requested script entrypoint exists."""

    assert Path("scripts/experiment_3805_external_research_refresh.py").exists()
