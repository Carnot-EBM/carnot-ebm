"""Tests for Exp6476 corpus label-commitment forensic.

Spec refs: REQ-VERIFY-6476, SCENARIO-VERIFY-6476-CAUSAL-COMMITMENT,
SCENARIO-VERIFY-6476-POSTHOC-ATTACKS, SCENARIO-VERIFY-6476-ROWS,
SCENARIO-VERIFY-6476-NO-MUTATION.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot import experiment_6476_v556_corpus_label_commitment_forensic as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _receipt(
    receipt_id: str,
    *,
    contains_label_hash: bool = False,
    contains_membership_hash: bool = False,
    immutable: bool = True,
    before_deadline: bool = True,
) -> dict[str, object]:
    return {
        "receipt_id": receipt_id,
        "receipt_kind": "fixture_signed_content_address",
        "contains_label_hash": contains_label_hash,
        "contains_membership_hash": contains_membership_hash,
        "immutable": immutable,
        "observed_before_first_inference": before_deadline,
        "content_bound_to_unit": True,
    }


def _with_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_verify_6476_spec_declares_fields_and_scenarios() -> None:
    """REQ-VERIFY-6476: OpenSpec owns the forensic contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6476") :]
    for marker in (
        "SCENARIO-VERIFY-6476-CAUSAL-COMMITMENT",
        "SCENARIO-VERIFY-6476-POSTHOC-ATTACKS",
        "SCENARIO-VERIFY-6476-ROWS",
        "SCENARIO-VERIFY-6476-NO-MUTATION",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_verify_6476_causal_commitment_is_conjunctive() -> None:
    """SCENARIO-VERIFY-6476-CAUSAL-COMMITMENT: labels and membership both predate inference."""

    row = mod.adjudicate_commitment_row(
        unit_id="held-1",
        partition="selection_held",
        label_hash="sha256:" + "1" * 64,
        membership_hash="sha256:" + "2" * 64,
        prompt_hashes=["sha256:" + "3" * 64],
        first_raw_event_time={"first_raw_event_mtime_ns": 1000},
        checkpoint_receipt={"path": "checkpoint.json"},
        file_receipts=[],
        label_receipts=[_receipt("label-seal", contains_label_hash=True)],
        membership_receipts=[_receipt("member-seal", contains_membership_hash=True)],
    )
    assert row["held_unit"] is True
    assert row["label_precommit_proof"] is True
    assert row["membership_precommit_proof"] is True
    assert row["creditable_for_salvage"] is True
    assert row["missing_or_posthoc_reasons"] == []

    label_only = mod.adjudicate_commitment_row(
        unit_id="held-2",
        partition="selection_held",
        label_hash="sha256:" + "1" * 64,
        membership_hash="sha256:" + "2" * 64,
        prompt_hashes=[],
        first_raw_event_time={"first_raw_event_mtime_ns": 1000},
        checkpoint_receipt={},
        file_receipts=[],
        label_receipts=[_receipt("label-only", contains_label_hash=True)],
        membership_receipts=[],
    )
    assert label_only["label_precommit_proof"] is True
    assert label_only["membership_precommit_proof"] is False
    assert label_only["creditable_for_salvage"] is False
    assert "missing_immutable_pre_inference_membership_proof" in label_only[
        "missing_or_posthoc_reasons"
    ]

    copied_timestamp = mod.adjudicate_commitment_row(
        unit_id="held-3",
        partition="audit_held",
        label_hash="sha256:" + "1" * 64,
        membership_hash="sha256:" + "2" * 64,
        prompt_hashes=[],
        first_raw_event_time={"first_raw_event_mtime_ns": 1000},
        checkpoint_receipt={},
        file_receipts=[],
        label_receipts=[
            _receipt(
                "copied-mtime",
                contains_label_hash=True,
                immutable=False,
                before_deadline=True,
            )
        ],
        membership_receipts=[
            _receipt(
                "post-git",
                contains_membership_hash=True,
                immutable=True,
                before_deadline=False,
            )
        ],
    )
    assert copied_timestamp["label_precommit_proof"] is False
    assert copied_timestamp["membership_precommit_proof"] is False
    assert copied_timestamp["creditable_for_salvage"] is False


def test_scenario_verify_6476_attacks_fail_closed() -> None:
    """SCENARIO-VERIFY-6476-POSTHOC-ATTACKS: attack fixtures do not salvage."""

    rows = [
        {
            "held_unit": True,
            "label_precommit_proof": False,
            "membership_precommit_proof": False,
            "creditable_for_salvage": False,
        }
    ]
    attacks = mod.build_attack_matrix(rows)
    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    for row in attacks["rows"]:
        assert row["detected"] is True
        assert row["accepted_as_precommit"] is False


def test_scenario_verify_6476_independent_replay_failure_branches() -> None:
    """SCENARIO-VERIFY-6476-ROWS: replay reports raw, label, and prompt defects."""

    manifest = mod._read_json(REPO / mod.EXP6463_MANIFEST)
    problem = manifest["problems"][0]
    artifact = mod._read_json(REPO / mod.EXP6463_RESULT)
    first_row = next(
        row
        for row in artifact["per_unit_rows"]["rows"]
        if row["unit_id"] == problem["problem_id"]
    )

    assert mod._utc_from_ns(None) is None
    assert mod._git_blob_map(REPO, []) == {}

    missing_raw = dict(first_row)
    missing_raw["raw_output_path"] = "missing/raw.bin"
    checks = mod._independent_unit_checks(root=REPO, problem=problem, rows=[missing_raw])
    assert checks["missing_raw_file_count"] == 1

    bad_label = dict(first_row)
    bad_label["observed_exact_label_sha256"] = "sha256:" + "0" * 64
    checks = mod._independent_unit_checks(root=REPO, problem=problem, rows=[bad_label])
    assert checks["exact_label_replay_mismatch_count"] == 1

    bad_prompt = dict(first_row)
    bad_prompt["prompt_sha256"] = "sha256:" + "0" * 64
    checks = mod._independent_unit_checks(root=REPO, problem=problem, rows=[bad_prompt])
    assert checks["prompt_template_mismatch_count"] == 1


def test_scenario_verify_6476_current_artifact_retires_lineage(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6476-ROWS: current Exp6463 evidence cannot salvage held labels."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.5,
        tests_run=_passing_tests(),
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_forensic_retire_lineage"
    assert artifact["corpus_lineage_disposition"] == "retire_lineage"
    assert artifact["corpus_label_commitment_salvage_score"] == 0.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["first_inference_event_receipt"]["unit_id"] == "exp6463-policy-00"
    assert artifact["first_inference_event_receipt"]["event_id"].startswith("evt-")
    assert artifact["first_inference_event_receipt"]["new_inference_performed"] is False
    assert artifact["preconditions_checked"]["new_labels_written"] is False
    assert artifact["preconditions_checked"]["new_membership_manifest_written"] is False
    assert artifact["protected_files_unchanged"]["unchanged"] is True

    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate["unit_count"] == 48
    assert aggregate["held_unit_count"] == 36
    assert aggregate["held_units_with_label_precommit_proof"] == 0
    assert aggregate["held_units_with_membership_precommit_proof"] == 0
    assert aggregate["score_from_rows"] == 0.0
    assert aggregate["disposition_from_rows"] == "retire_lineage"
    assert aggregate["matches_reported"] is True
    assert len(artifact["missing_or_posthoc_proof_rows"]) == 36
    assert mod.validate_artifact(artifact) == []

    result_path = tmp_path / "experiment_6476.json"
    mod.write_artifact(artifact, result_path)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_6476_validation_and_cli_edges(
    tmp_path: Path,
    capsys,
) -> None:
    """SCENARIO-VERIFY-6476-NO-MUTATION: validation blocks forged summaries."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.5,
        tests_run=_passing_tests(),
    )

    bad = deepcopy(artifact)
    del bad["status"]
    assert "missing required field: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["corpus_label_commitment_salvage_score"] = 1.0
    assert "salvage score requires every held proof" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["corpus_label_commitment_salvage_score"] = 0.5
    assert "corpus_label_commitment_salvage_score mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["corpus_lineage_disposition"] = "development_only"
    assert "corpus_lineage_disposition mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["attack_matrix"]["all_attacks_fail_closed"] = False
    assert "attack matrix must fail closed" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["preconditions_checked"]["new_inference_performed"] = True
    assert "new inference is forbidden" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["preconditions_checked"]["new_labels_written"] = True
    assert "new labels are forbidden" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["preconditions_checked"]["new_membership_manifest_written"] = True
    assert "new membership manifest is forbidden" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["preconditions_checked"]["timestamps_repaired"] = True
    assert "timestamp repair is forbidden" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    assert "verifier_is_oracle must be true for recorded hash and causal-order checks" in (
        mod.validate_artifact(_with_checksum(bad))
    )

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    assert "missing field_principles entry: status" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "retire_lineage"
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    result_path = tmp_path / "experiment_6476_cli.json"
    assert mod.main(["--date", "20260821", "--result-path", str(result_path)]) == 0
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["corpus_lineage_disposition"] == "retire_lineage"

    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    validate_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert validate_out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    missing_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert missing_out["ok"] is False
    assert missing_out["errors"] == ["artifact missing"]
