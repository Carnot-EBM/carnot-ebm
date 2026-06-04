"""Tests for Exp 3792 product-headline G4 provenance confirmation.

Spec: REQ-PUBLISH-3792, SCENARIO-PUBLISH-3792
"""

from __future__ import annotations

import json
import builtins
from pathlib import Path

from scripts import experiment_3792_product_headline_provenance_confirmation_g4 as exp3792


ROOT = Path(__file__).resolve().parents[2]


def test_actual_artifacts_confirm_numbers_and_g4_split():
    """Actual checked-in artifacts produce the honest G4 split.

    Spec: REQ-PUBLISH-3792, SCENARIO-PUBLISH-3792
    """
    artifact = exp3792.build_artifact(ROOT)
    rows = {row["number"]: row for row in artifact["provenance_table"]}

    exp1999 = rows["exp1999_humaneval_repair_0.66_to_0.84"]
    assert exp1999["source_artifact"].endswith("results/experiment_1999_code_verification_humaneval.json")
    assert Path(exp1999["source_artifact"]).is_absolute()
    assert exp1999["n"] == 50
    assert exp1999["headline_numbers_match_north_star"] is True
    assert exp1999["seed_present"] is False
    assert exp1999["checksum_present"] is False
    assert exp1999["g4_pass"] is False
    assert "missing random_seed" in exp1999["caveat"]

    exp2090 = rows["exp2090_crane_rigid_0.70_to_crane_0.85"]
    assert exp2090["source_artifact"].endswith("results/experiment_2090_crane_humaneval.json")
    assert Path(exp2090["source_artifact"]).is_absolute()
    assert exp2090["n"] == 50
    assert exp2090["headline_numbers_match_north_star"] is True
    assert exp2090["seed_present"] is True
    assert exp2090["checksum_present"] is True
    assert exp2090["g4_pass"] is True

    assert artifact["exp1999_g4_pass"] is False
    assert artifact["exp2090_g4_pass"] is True
    assert artifact["product_headline_restorable"] == "not_yet_headline_eligible"
    assert artifact["operator_curated_doc_unedited"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["random_seed"] == {"exp1999": None, "exp2090": 42}
    assert artifact["reproducibility_checksum"]["exp1999"] is None
    assert artifact["reproducibility_checksum"]["exp2090"] == "bfb0acdb53773a49"
    assert len(artifact["cited_upstream_artifacts"]) == 3
    assert any(path.endswith("results/experiment_227_results.json") for path in artifact["cited_upstream_artifacts"])
    for marker in ("GG" + "UF", "CU" + "DA"):
        assert marker not in json.dumps(artifact)
    assert artifact["honest_verdict"] == (
        "complete: product_headline_provenance_confirmed_exp1999_g4_false_"
        "exp2090_g4_true_headline_not_yet_eligible_operator_curated_doc_unedited"
    )


def test_missing_primary_artifact_records_g4_block_without_fabrication(tmp_path):
    """Missing product artifacts are recorded as blocked, not inferred.

    Spec: REQ-PUBLISH-3792
    """
    (tmp_path / "results").mkdir()
    row = exp3792.build_provenance_row(
        root=tmp_path,
        number="exp1999_humaneval_repair_0.66_to_0.84",
        relative_path=Path("results/missing_exp1999.json"),
        before_key="baseline_pass_rate",
        after_key="repair_pass_rate",
        expected_before=0.66,
        expected_after=0.84,
        n_key="dataset_size",
    )

    assert row["source_artifact"] == str((tmp_path / "results/missing_exp1999.json").resolve())
    assert row["n"] is None
    assert row["headline_numbers_match_north_star"] is False
    assert row["seed_present"] is False
    assert row["checksum_present"] is False
    assert row["g4_pass"] is False
    assert row["caveat"] == "artifact_not_found_cannot_confirm_g4"


def test_discrepant_numbers_fail_g4_even_with_seed_and_checksum(tmp_path):
    """A seeded artifact with the wrong numbers must not be headline-confirmed.

    Spec: REQ-PUBLISH-3792, SCENARIO-PUBLISH-3792
    """
    path = tmp_path / "results" / "experiment_1999_code_verification_humaneval.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "dataset_size": 50,
                "baseline_pass_rate": 0.65,
                "repair_pass_rate": 0.84,
                "random_seed": 1999,
                "reproducibility_checksum": "abc123",
                "inference_substrate": "archival_json_fixture",
            }
        )
    )

    row = exp3792.build_provenance_row(
        root=tmp_path,
        number="exp1999_humaneval_repair_0.66_to_0.84",
        relative_path=Path("results/experiment_1999_code_verification_humaneval.json"),
        before_key="baseline_pass_rate",
        after_key="repair_pass_rate",
        expected_before=0.66,
        expected_after=0.84,
        n_key="dataset_size",
    )

    assert row["headline_numbers_match_north_star"] is False
    assert row["seed_present"] is True
    assert row["checksum_present"] is True
    assert row["g4_pass"] is False
    assert "number discrepancy" in row["caveat"]


def test_common_n_and_metadata_substrate_fallbacks(tmp_path):
    """Secondary metadata fields are allowed only as source-artifact fields.

    Spec: REQ-PUBLISH-3792
    """
    path = tmp_path / "results" / "experiment_2090_crane_humaneval.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "sample_size": 50,
                "rigid_pass_rate": 0.70,
                "crane_pass_rate": 0.85,
                "random_seed": 42,
                "reproducibility_checksum": "bfb0acdb53773a49",
                "metadata": {"inference_mode": "archival_replay"},
            }
        )
    )

    row = exp3792.build_provenance_row(
        root=tmp_path,
        number="exp2090_crane_rigid_0.70_to_crane_0.85",
        relative_path=Path("results/experiment_2090_crane_humaneval.json"),
        before_key="rigid_pass_rate",
        after_key="crane_pass_rate",
        expected_before=0.70,
        expected_after=0.85,
    )

    assert row["n"] == 50
    assert row["substrate"] == "archival_replay"
    assert row["g4_pass"] is True


def test_exp227_missing_and_headline_status_helpers(tmp_path):
    """Refuted contrast absence and status helper branches are explicit.

    Spec: REQ-PUBLISH-3792
    """
    missing = exp3792._exp227_contrast(tmp_path)
    assert missing == {
        "source_artifact": str((tmp_path / "results" / "experiment_227_results.json").resolve()),
        "available": False,
    }

    clean_rows = [{"g4_pass": True, "caveat": "none"}, {"g4_pass": True, "caveat": "none"}]
    caveat_rows = [
        {"g4_pass": True, "caveat": "none"},
        {"g4_pass": True, "caveat": "still needs operator review"},
    ]
    assert exp3792._headline_status(clean_rows) == "restorable"
    assert exp3792._headline_status(caveat_rows) == "restorable_with_caveat"
    assert exp3792._terminal_status("restorable") == "restorable"


def test_main_writes_artifact_to_results(monkeypatch, tmp_path):
    """The CLI path writes the required deliverable JSON.

    Spec: REQ-PUBLISH-3792, SCENARIO-PUBLISH-3792
    """
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_1999_code_verification_humaneval.json").write_text(
        json.dumps(
            {
                "dataset_size": 50,
                "baseline_pass_rate": 0.66,
                "repair_pass_rate": 0.84,
            }
        )
    )
    (results / "experiment_2090_crane_humaneval.json").write_text(
        json.dumps(
            {
                "duration_s": 1.25,
                "random_seed": 42,
                "reproducibility_checksum": "bfb0acdb53773a49",
                "crane_pass_rate": 0.85,
                "rigid_pass_rate": 0.70,
            }
        )
    )
    (results / "experiment_227_results.json").write_text(
        json.dumps(
            {
                "metadata": {"sample_size": 30},
                "statistics": {
                    "baseline": {"pass_at_1": 0.23333333333333334},
                    "verify_repair": {"pass_at_1": 0.23333333333333334, "n_repaired": 0},
                    "improvement": {"delta": 0.0},
                },
            }
        )
    )

    monkeypatch.setattr(exp3792, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        exp3792,
        "ARTIFACT_PATH",
        results / "experiment_3792_product_headline_provenance_confirmation_g4.json",
    )

    assert exp3792.main() == 0
    loaded = json.loads(exp3792.ARTIFACT_PATH.read_text())
    assert loaded["honest_verdict"].startswith("complete: product_headline_provenance_confirmed")
    assert loaded["operator_curated_doc_unedited"] is True


def test_main_blocks_when_yaml_import_is_unavailable(monkeypatch, tmp_path):
    """Interpreter preconditions block without pretending G4 passed.

    Spec: REQ-PUBLISH-3792
    """
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("yaml unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(exp3792, "ARTIFACT_PATH", tmp_path / "results" / "blocked.json")

    assert exp3792.main() == 1
    blocked = json.loads(exp3792.ARTIFACT_PATH.read_text())
    assert blocked["honest_verdict"] == "blocked_interpreter_yaml_unavailable"
    assert blocked["exp1999_g4_pass"] is False
    assert blocked["exp2090_g4_pass"] is False
