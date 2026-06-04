"""Tests for Exp 3816 external research refresh.

Spec refs: REQ-REPORT-3816, SCENARIO-REPORT-3816.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import external_research_refresh_3816 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

PRIOR_REFERENCES = """# Research References

## .348 additions - early-rejection predictive verification, outcome-guided process rewards, EDLM impl confirmation (2026-06-04)

- Existing .348 content stays untouched.

## .349 additions - geometry-calibrated conformal abstention, selective conformal judging, and the EDLM reasoning datapoint (2026-06-04)

Added by the `.349 planning sweep (Claude Opus 4.8). Append-only into the converged record: `paper_ready` stays TRUE (G1-G4, confirmed via publication_gate.py), FoVer 0.9131 stays frozen, both energy routes stay bounded per `[[project_energy_selection_thesis_bounded]]` / `[[project_thesis_a_ebt_seeded]]`, the verifier-moat thread stays closed per `[[reference_deep_think_post_bounded_2026_06]]`, and EDLM stays an operator-seeded route (preflight returned GO at `.347 exp3793; the loop does NOT self-seed -- the P3 Verification Trap). `.349 is a LEAN, NON-CHURN maintenance milestone: it WIRES the now-usable Anomaly-Escalation classifier as a recommend-only advisory hook (exp3802 reached false-escalation 0.0 / recall 1.0), REPAIRS the blocked HTTP/REST abstention surface (exp3801 blocked_http_abstention_e2e_failed), records the product-headline status HONESTLY (both candidate positives fail provenance: exp1999 re-run delta=0.0 per exp3798; exp2090 CRANE flags CRITICAL on live adversarial re-check), continues Tier-3 self-learning, and STAGES the EDLM seed for the operator.

Numbers are source-reported by the papers. They are not Carnot measurements and must be independently re-derived before entering any forward-facing claim.

- **arXiv:2604.27914 - "Geometry-Calibrated Conformal Abstention for Language Models" (2026; arXiv resolved):** Track: selective-prediction / abstention surface. Existing anchor stays.
- **arXiv:2602.13110 - "SCOPE: Selective Conformal Optimized Pairwise LLM Judging" (2026; arXiv resolved):** Track: selective-prediction / abstention surface. Existing anchor stays.
- **EDLM reasoning datapoint -- Dream 7B as an EDLM variant (via arXiv:2410.21357 family; source pages, 2026):** Track: EDLM operator-seed. Existing anchor stays.
"""


def _seed_references(root: Path, text: str = PRIOR_REFERENCES) -> str:
    path = root / mod.RESEARCH_REFERENCES_REL_PATH
    path.write_text(text, encoding="utf-8")
    return text


def test_req_report_3816_spec_anchor_exists() -> None:
    """REQ-REPORT-3816: OpenSpec declares the refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3816" in spec
    assert "SCENARIO-REPORT-3816" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.RESEARCH_REFERENCES_REL_PATH.as_posix() in spec
    for arxiv_id in mod.REFERENCE_IDS:
        assert arxiv_id in spec
    for excluded_id in mod.EXCLUDED_DUPLICATE_IDS:
        assert excluded_id in spec


def test_scenario_report_3816_confirms_349_and_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3816: the .349 section is confirmed and extended."""

    before = _seed_references(tmp_path)

    out_path = mod.run(tmp_path, started_s=100.0, now_s=100.25)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    after = (tmp_path / mod.RESEARCH_REFERENCES_REL_PATH).read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert after.startswith(before)
    assert after.count(mod.SECTION_HEADER) == 1
    assert after.count("arXiv:2604.27914") == 1
    assert after.count("arXiv:2602.13110") == 1
    assert after.count("arXiv:2410.21357") == 1
    assert "arXiv:2604.13991" not in mod.REFERENCE_IDS
    assert "arXiv:2507.02092" not in mod.REFERENCE_IDS
    assert "Numbers are source-reported by the papers" in after
    assert "reference_deep_think_post_bounded_2026_06" in after
    assert "project_energy_selection_thesis_bounded" in after
    assert "project_thesis_a_ebt_seeded" in after
    for arxiv_id in mod.REFERENCE_IDS:
        assert arxiv_id in after
    for track in (
        "selective-prediction / abstention surface",
        "EDLM operator-seed",
        "Tier-3 fast-path process verification",
    ):
        assert track in after

    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["references_section_intact"] is True
    assert artifact["references_added"] == list(mod.REFERENCE_IDS)
    assert artifact["n_references_added"] == len(mod.REFERENCE_IDS)
    assert artifact["section_appended_not_replaced"] is True
    assert artifact["numbers_are_as_reported"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == 0.25
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["section_sha256"] == mod.sha256_text(mod.extract_349_section(after))
    assert artifact["adversarial_verify_clean"] is True
    assert {row["arxiv_id"] for row in artifact["references"]} == set(mod.REFERENCE_IDS)
    assert all(row["arxiv_abs_resolved"] is True for row in artifact["references"])
    assert all(row["numbers_are_as_reported"] is True for row in artifact["references"])
    assert all(row["source_kind"] == "arxiv_abs" for row in artifact["references"])

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "live-model" not in encoded


def test_req_report_3816_append_is_idempotent(tmp_path: Path) -> None:
    """REQ-REPORT-3816: a rerun does not duplicate the .349 filings."""

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


def test_req_report_3816_section_integrity_is_checked(tmp_path: Path) -> None:
    """REQ-REPORT-3816: missing .349 anchors fail before appending."""

    path = tmp_path / mod.RESEARCH_REFERENCES_REL_PATH
    path.write_text("# Research References\n\n## .349 additions - incomplete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="intact"):
        mod.append_references_if_missing(path)

    with pytest.raises(ValueError, match="section"):
        mod.extract_349_section("# Research References\n")

    section_plus_next = PRIOR_REFERENCES + "\n## .350 additions - future\n\n- untouched\n"
    extracted = mod.extract_349_section(section_plus_next)
    assert "## .350 additions" not in extracted

    with pytest.raises(ValueError, match="missing"):
        mod.confirm_349_section_intact(PRIOR_REFERENCES.replace("arXiv:2602.13110", ""))


def test_req_report_3816_helper_fallbacks_are_explicit() -> None:
    """REQ-REPORT-3816: verification-report helper fallbacks are deterministic."""

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
            lambda artifact: artifact.update(references_section_intact=False),
            "references_section_intact",
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
def test_req_report_3816_validation_rejects_schema_violations(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-REPORT-3816: validation enforces the required artifact fields."""

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


def test_scenario_report_3816_script_entrypoint_exists() -> None:
    """SCENARIO-REPORT-3816: the requested script entrypoint exists."""

    assert Path("scripts/experiment_3816_external_research_refresh_v349.py").exists()
