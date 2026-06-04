"""Tests for Exp 3782 technical-report G4 correction prep.

Spec refs: REQ-REPORT-3782, SCENARIO-REPORT-3782.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import technical_report_g4_correction_prep_3782 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


OLD_TRAJECTORY = (
    "The trajectory of this project is: we tried the obvious approach, pivoted "
    "to encoding external knowledge as formal constraints, proved that code "
    "verification (+3.0pp HumanEval) and typed constraint verification (+4.9pp) "
    "work on inference artifacts, and closed the milestone."
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_repo(root: Path) -> dict[str, str]:
    (root / "docs" / "research-notes").mkdir(parents=True, exist_ok=True)
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "technical-report.md").write_text(
        "# Technical Report\n\n### The story\n\n" + OLD_TRAJECTORY + "\n",
        encoding="utf-8",
    )
    (root / "ops" / "north-star.md").write_text(
        'OPERATOR ACTION: replace "8%->80%"/"0%->36%"/"+3.0pp"; '
        "Exp 227 shows delta=0.0 and n_repaired=0.\n",
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "experiment_227_results.json",
        {
            "experiment": 227,
            "metadata": {"sample_size": 30, "run_seed": 227},
            "statistics": {
                "baseline": {"pass_at_1": 0.23333333333333334},
                "verify_repair": {
                    "pass_at_1": 0.23333333333333334,
                    "n_repaired": 0,
                },
                "improvement": {"delta": 0.0},
                "repair_stats": {"n_repaired": 0},
            },
        },
    )
    _write_json(
        root / "results" / "experiment_1999_code_verification_humaneval.json",
        {
            "experiment_id": 1999,
            "baseline_pass_rate": 0.66,
            "repair_pass_rate": 0.84,
            "dataset_size": 50,
            "honest_verdict": "ising_guided_fuzzing_implemented",
        },
    )
    _write_json(
        root / "results" / "experiment_2090_crane_humaneval.json",
        {
            "experiment": 2090,
            "rigid_pass_rate": 0.70,
            "crane_pass_rate": 0.85,
            "pass_rate_delta": 0.15,
            "random_seed": 42,
            "reproducibility_checksum": "bfb0acdb53773a49",
            "honest_verdict": "CRANE evaluated vs rigid grammar on 50 HumanEval problems.",
        },
    )
    return {
        "technical_report": (root / "docs" / "technical-report.md").read_text(
            encoding="utf-8"
        ),
        "north_star": (root / "ops" / "north-star.md").read_text(encoding="utf-8"),
    }


def test_req_report_3782_spec_anchor_exists() -> None:
    """REQ-REPORT-3782: OpenSpec declares the correction-prep contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3782" in spec
    assert "SCENARIO-REPORT-3782" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.PROPOSAL_REL_PATH.as_posix() in spec


def test_scenario_report_3782_prepares_correction_without_editing_report(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3782: proposal and artifact are written, report is not edited."""

    before_docs = _seed_repo(tmp_path)

    out_path = mod.run(tmp_path, started_s=10.0, now_s=10.25)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    proposal = (tmp_path / mod.PROPOSAL_REL_PATH).read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.TERMINAL_VERDICT
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["proposed_correction_written"] is True
    assert artifact["operator_curated_doc_unedited"] is True
    assert artifact["duration_s"] == 0.25
    assert artifact["random_seed"] == 3782
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    unsupported = artifact["unsupported_numbers_identified"]
    assert unsupported["refuted_prose_numbers"] == ["8%->80%", "0%->36%", "+3.0pp"]
    assert unsupported["exp227"]["improvement_delta"] == 0.0
    assert unsupported["exp227"]["n_repaired"] == 0

    confirmed = {row["experiment_id"]: row for row in artifact["real_numbers_confirmed"]}
    assert confirmed[1999]["baseline"] == 0.66
    assert confirmed[1999]["improved"] == 0.84
    assert confirmed[1999]["delta_pp"] == 18.0
    assert confirmed[1999]["n"] == 50
    assert confirmed[1999]["g4_passes"] is False
    assert "random_seed" in confirmed[1999]["g4_missing_fields"]
    assert "reproducibility_checksum" in confirmed[1999]["g4_missing_fields"]
    assert confirmed[2090]["baseline"] == 0.70
    assert confirmed[2090]["improved"] == 0.85
    assert confirmed[2090]["delta_pp"] == 15.0
    assert confirmed[2090]["n"] == 50
    assert confirmed[2090]["g4_passes"] is False
    assert confirmed[2090]["g4_missing_fields"] == ["structured_n"]

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert set(cited) == {227, 1999, 2090}
    assert cited[227]["path"] == "results/experiment_227_results.json"
    assert "statistics.improvement.delta" in cited[227]["fields_imported"]
    assert "baseline_pass_rate" in cited[1999]["fields_imported"]
    assert "crane_pass_rate" in cited[2090]["fields_imported"]
    assert all(len(row["sha256"]) == 64 for row in cited.values())

    assert "OPERATOR ACTION Proposal" in proposal
    assert OLD_TRAJECTORY in proposal
    assert "+3.0pp HumanEval" in proposal
    assert "Exp 227 reports 0.0pp delta and 0 repaired cases" in proposal
    assert "Exp 1999 reports 0.66 -> 0.84 over n=50" in proposal
    assert "Exp 2090 reports 0.70 -> 0.85 over n=50" in proposal
    assert "```diff" in proposal

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "unsloth/" not in encoded
    assert "gemma-4-" not in encoded
    assert "Qwen3.5-" not in encoded

    assert (tmp_path / "docs" / "technical-report.md").read_text(
        encoding="utf-8"
    ) == before_docs["technical_report"]
    assert (tmp_path / "ops" / "north-star.md").read_text(
        encoding="utf-8"
    ) == before_docs["north_star"]


def test_req_report_3782_validation_rejects_curated_doc_edit(tmp_path: Path) -> None:
    """REQ-REPORT-3782: validation blocks silent operator-curated doc edits."""

    before_docs = _seed_repo(tmp_path)
    artifact = mod.build_artifact(tmp_path, before_docs, started_s=1.0, now_s=1.1)
    artifact["operator_curated_doc_unedited"] = False

    try:
        mod.validate_artifact(artifact)
    except ValueError as exc:
        assert "operator_curated_doc_unedited" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validate_artifact accepted an edited curated doc")


def test_req_report_3782_helper_fallbacks_and_error_paths(tmp_path: Path) -> None:
    """REQ-REPORT-3782: defensive parsing paths stay explicit."""

    before_docs = _seed_repo(tmp_path)

    assert mod.extract_n({"metadata": {"sample_size": 12}}) == (
        12,
        "metadata.sample_size",
        True,
    )
    assert mod.extract_n({"cohort": {"case_count": 13}}) == (
        13,
        "cohort.case_count",
        True,
    )
    assert mod.extract_n({}) == (None, "missing", False)
    assert mod.report_is_clean(None) is True
    assert mod.severity_rank("surprise") == -1

    try:
        mod.find_trajectory_paragraph("no trajectory here")
    except ValueError as exc:
        assert "trajectory paragraph" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing trajectory paragraph was accepted")

    try:
        mod.find_old_clause("The trajectory of this project is: no old clause.")
    except ValueError as exc:
        assert "old code-repair" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing old clause was accepted")

    assert mod.operator_curated_docs_unchanged(tmp_path, {}) is False
    (tmp_path / "docs" / "technical-report.md").write_text("edited\n", encoding="utf-8")
    assert mod.operator_curated_docs_unchanged(tmp_path, before_docs) is False


def test_scenario_report_3782_script_entrypoint_exists() -> None:
    """SCENARIO-REPORT-3782: the requested script entrypoint exists."""

    assert Path("scripts/experiment_3782_technical_report_g4_correction_prep.py").exists()
