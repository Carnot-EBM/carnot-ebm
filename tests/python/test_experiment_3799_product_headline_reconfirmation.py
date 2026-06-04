"""Tests for Exp 3799 product-headline provenance reconfirmation.

Spec: REQ-PUBLISH-3799, SCENARIO-PUBLISH-3799
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

from scripts import experiment_3799_product_headline_provenance_reconfirmation as exp3799


ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_actual_artifacts_reconfirm_g4_but_keep_headline_demoted():
    """Actual checked-in artifacts preserve the Exp 3798 caveat.

    Spec: REQ-PUBLISH-3799, SCENARIO-PUBLISH-3799
    """
    artifact = exp3799.build_artifact(ROOT)
    rows = {row["number"]: row for row in artifact["provenance_table"]}

    rerun = rows["exp3798_rerun_code_repair_baseline_0.13_repair_0.13_delta_0.0pp"]
    assert rerun["source_artifact"].endswith(
        "results/experiment_3798_g4_product_headline_restoration.json"
    )
    assert Path(rerun["source_artifact"]).is_absolute()
    assert rerun["n"] == 30
    assert rerun["seed_present"] is True
    assert rerun["checksum_present"] is True
    assert rerun["positive_control_passed"] is True
    assert rerun["g4_provenance_complete"] is True
    assert rerun["g4_pass"] is True
    assert rerun["baseline_pass1"] == 0.13333333333333333
    assert rerun["repair_pass1"] == 0.13333333333333333
    assert rerun["repair_delta_pp"] == 0.0
    assert "upstream_flagged_adversarial" in rerun["caveat"]
    assert "zero_delta_headline_stays_demoted" in rerun["caveat"]

    crane = rows["exp2090_crane_rigid_0.70_to_crane_0.85"]
    assert crane["source_artifact"].endswith("results/experiment_2090_crane_humaneval.json")
    assert Path(crane["source_artifact"]).is_absolute()
    assert crane["n"] == 50
    assert crane["seed_present"] is True
    assert crane["checksum_present"] is True
    assert crane["positive_control_passed"] is None
    assert crane["g4_pass"] is True

    assert artifact["rerun_code_repair_g4_pass"] is True
    assert artifact["exp2090_g4_pass"] is True
    assert artifact["product_headline_restorable"] == "not_yet_headline_eligible"
    assert artifact["operator_curated_doc_unedited"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["random_seed"] == {"exp3798": 42, "exp2090": 42}
    assert artifact["reproducibility_checksum"] == {
        "exp3798": "a854c6f82908fec3",
        "exp2090": "bfb0acdb53773a49",
    }
    assert len(artifact["cited_upstream_artifacts"]) == 2
    assert all(Path(path).is_absolute() for path in artifact["cited_upstream_artifacts"])
    for marker in ("GG" + "UF", "CU" + "DA", "unsloth/"):
        assert marker not in json.dumps(artifact)
    assert artifact["honest_verdict"] == (
        "complete: product_headline_provenance_reconfirmed_rerun_g4_true_"
        "exp2090_g4_true_headline_not_yet_eligible_operator_curated_doc_unedited"
    )


def test_missing_exp3798_records_graceful_block_without_fabrication(tmp_path):
    """Missing rerun artifact records the gate miss instead of fabricating G4.

    Spec: REQ-PUBLISH-3799
    """
    results = tmp_path / "results"
    _write_json(
        results / "experiment_2090_crane_humaneval.json",
        {
            "honest_verdict": "CRANE evaluated vs rigid grammar on 50 HumanEval problems.",
            "rigid_pass_rate": 0.70,
            "crane_pass_rate": 0.85,
            "pass_rate_delta": 0.15,
            "random_seed": 42,
            "reproducibility_checksum": "bfb0acdb53773a49",
        },
    )

    artifact = exp3799.build_artifact(tmp_path)
    rows = {row["number"]: row for row in artifact["provenance_table"]}
    rerun = rows["exp3798_rerun_code_repair_baseline_0.13_repair_0.13_delta_0.0pp"]

    assert artifact["honest_verdict"] == (
        "blocked: exp3798_did_not_produce_clean_artifact_headline_stays_demoted"
    )
    assert rerun["source_artifact"] == str(
        (results / "experiment_3798_g4_product_headline_restoration.json").resolve()
    )
    assert rerun["g4_pass"] is False
    assert rerun["caveat"] == "artifact_not_found_cannot_confirm_g4"
    assert artifact["rerun_code_repair_g4_pass"] is False
    assert artifact["exp2090_g4_pass"] is True
    assert artifact["product_headline_restorable"] == "not_yet_headline_eligible"
    assert artifact["random_seed"]["exp3798"] is None
    assert artifact["reproducibility_checksum"]["exp3798"] is None


def test_rerun_g4_requires_positive_control_complete_provenance_and_nontrivial_n(tmp_path):
    """The rerun number fails G4 when any Exp 3798 gate input is missing.

    Spec: REQ-PUBLISH-3799
    """
    path = tmp_path / "results" / "experiment_3798_g4_product_headline_restoration.json"
    base = {
        "baseline_pass1": 0.13,
        "repair_pass1": 0.31,
        "repair_delta_pp": 18.0,
        "random_seed": 42,
        "reproducibility_checksum": "checksum",
        "n": 30,
        "positive_control_passed": True,
        "g4_provenance_complete": True,
        "inference_substrate": "live_llm_inference",
    }

    for key, expected_caveat in (
        ("random_seed", "missing_random_seed"),
        ("reproducibility_checksum", "missing_reproducibility_checksum"),
        ("n", "missing_non_trivial_n"),
        ("positive_control_passed", "positive_control_failed"),
        ("g4_provenance_complete", "g4_provenance_incomplete"),
    ):
        payload = dict(base)
        if key == "positive_control_passed":
            payload[key] = False
        elif key == "g4_provenance_complete":
            payload[key] = False
        else:
            payload.pop(key)
        _write_json(path, payload)

        row = exp3799.build_rerun_code_repair_row(tmp_path)
        assert row["g4_pass"] is False
        assert expected_caveat in row["caveat"]


def test_headline_status_helper_branches_are_explicit():
    """Headline status distinguishes clean, caveated, and demoted rows.

    Spec: REQ-PUBLISH-3799
    """
    clean_rerun = {"g4_pass": True, "caveat": "none"}
    clean_crane = {"g4_pass": True, "caveat": "none"}
    caveated_crane = {"g4_pass": True, "caveat": "already_G4_passing_per_exp3792_reasserted"}
    demoted_rerun = {"g4_pass": True, "caveat": "zero_delta_headline_stays_demoted"}

    assert exp3799.headline_status(clean_rerun, clean_crane) == "restorable"
    assert exp3799.headline_status(clean_rerun, caveated_crane) == "restorable_with_caveat"
    assert exp3799.headline_status(demoted_rerun, clean_crane) == "not_yet_headline_eligible"
    assert (
        exp3799.headline_status(
            {"g4_pass": True, "caveat": "upstream_flagged_adversarial"},
            clean_crane,
        )
        == "not_yet_headline_eligible"
    )
    assert exp3799.terminal_status("not_yet_headline_eligible") == "not_yet_eligible"


def test_exp2090_missing_and_fallback_fields_are_explicit(tmp_path):
    """Exp 2090 fallbacks stay explicit and sanitized.

    Spec: REQ-PUBLISH-3799
    """
    missing = exp3799.build_exp2090_crane_row(tmp_path)
    assert missing["g4_pass"] is False
    assert missing["caveat"] == "artifact_not_found_cannot_confirm_g4"

    path = tmp_path / "results" / "experiment_2090_crane_humaneval.json"
    _write_json(
        path,
        {
            "sample_size": 50,
            "rigid_pass_rate": 0.69,
            "crane_pass_rate": 0.85,
            "inference_mode": "archival_replay",
        },
    )
    discrepant = exp3799.build_exp2090_crane_row(tmp_path)
    assert discrepant["n"] == 50
    assert discrepant["substrate"] == "archival_replay"
    assert discrepant["g4_pass"] is False
    assert "missing_random_seed" in discrepant["caveat"]
    assert "missing_reproducibility_checksum" in discrepant["caveat"]
    assert "number_discrepancy_vs_exp3792" in discrepant["caveat"]

    _write_json(
        path,
        {
            "rigid_pass_rate": 0.70,
            "crane_pass_rate": 0.85,
            "random_seed": 42,
            "reproducibility_checksum": "checksum",
            "metadata": {"inference_mode": "metadata_replay"},
        },
    )
    no_n = exp3799.build_exp2090_crane_row(tmp_path)
    assert no_n["substrate"] == "metadata_replay"
    assert no_n["g4_pass"] is False
    assert "missing_non_trivial_n" in no_n["caveat"]


def test_main_writes_artifact_and_interpreter_block_is_honest(monkeypatch, tmp_path):
    """The CLI writes the deliverable and blocks honestly without yaml.

    Spec: REQ-PUBLISH-3799, SCENARIO-PUBLISH-3799
    """
    results = tmp_path / "results"
    _write_json(
        results / "experiment_3798_g4_product_headline_restoration.json",
        {
            "baseline_pass1": 0.13,
            "repair_pass1": 0.31,
            "repair_delta_pp": 18.0,
            "random_seed": 42,
            "reproducibility_checksum": "checksum3798",
            "n": 30,
            "positive_control_passed": True,
            "g4_provenance_complete": True,
            "inference_substrate": "live_llm_inference",
        },
    )
    _write_json(
        results / "experiment_2090_crane_humaneval.json",
        {
            "honest_verdict": "CRANE evaluated vs rigid grammar on 50 HumanEval problems.",
            "rigid_pass_rate": 0.70,
            "crane_pass_rate": 0.85,
            "pass_rate_delta": 0.15,
            "random_seed": 42,
            "reproducibility_checksum": "checksum2090",
        },
    )

    monkeypatch.setattr(exp3799, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        exp3799,
        "ARTIFACT_PATH",
        results / "experiment_3799_product_headline_provenance_reconfirmation.json",
    )

    assert exp3799.main() == 0
    loaded = json.loads(exp3799.ARTIFACT_PATH.read_text())
    assert loaded["honest_verdict"].startswith(
        "complete: product_headline_provenance_reconfirmed"
    )
    assert loaded["operator_curated_doc_unedited"] is True

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("yaml unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(exp3799, "ARTIFACT_PATH", results / "blocked.json")

    assert exp3799.main() == 1
    blocked = json.loads(exp3799.ARTIFACT_PATH.read_text())
    assert blocked["honest_verdict"] == "blocked_interpreter_yaml_unavailable"
    assert blocked["rerun_code_repair_g4_pass"] is False
    assert blocked["exp2090_g4_pass"] is False
