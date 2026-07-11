"""Tests for Exp5566 exact ASP/FSM near-miss corpus.

Spec refs: REQ-VERIFY-5566, SCENARIO-VERIFY-5566.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5541_llm_fsm_exact_fixture as fsm_mod
from carnot import experiment_5555_asp_fsm_nonmonotonic_fixture as asp_mod
from carnot import experiment_5566_exact_asp_fsm_near_miss_corpus as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5566_exact_asp_fsm_near_miss_corpus.py")


def _ready_upstream() -> dict[str, object]:
    fsm_artifact = fsm_mod.build_artifact(
        tests_run=[
            {
                "command": "tests/python/test_experiment_5541_llm_fsm_exact_fixture.py",
                "outcome": "passed",
            }
        ]
    )
    return asp_mod.build_artifact(
        upstream_artifact=fsm_artifact,
        tests_run=[
            {
                "command": "tests/python/test_experiment_5555_asp_fsm_nonmonotonic_fixture.py",
                "outcome": "passed",
            }
        ],
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_req_verify_5566_spec_declares_exact_near_miss_corpus_contract() -> None:
    """REQ-VERIFY-5566: OpenSpec anchors exact corpus gates and no-LLM labeling."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5566") : spec.index("### REQ-VERIFY-5501")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5566" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.CORPUS_RELATIVE_PATH) in section
    assert str(asp_mod.RESULT_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "`exact_validator_is_oracle` SHALL be `true`" in section
    assert "`llm_invoked` SHALL be `false`" in section
    assert "SHALL NOT use Exp 5552's missing grammar backend" in section
    assert "at least 120 rows" in normalized
    assert "at least 30 exact-validated rows" in normalized
    for family in mod.REQUIRED_FAMILIES:
        assert family.replace("_", "-") in normalized.replace("_", "-")
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5566_run_writes_balanced_leak_free_corpus(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5566: ready corpus requires balance, hashes, and controls."""

    artifact = mod.run(
        repo_root=tmp_path,
        upstream_artifact=_ready_upstream(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    corpus_path = tmp_path / mod.CORPUS_RELATIVE_PATH
    written = json.loads(result_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(corpus_path)

    assert written == artifact
    assert artifact["corpus_path"] == mod.CORPUS_RELATIVE_PATH.as_posix()
    assert artifact["corpus_sha256"] == mod.sha256_file(corpus_path)
    assert artifact["source_fixture_path"] == asp_mod.RESULT_RELATIVE_PATH.as_posix()
    assert artifact["exact_validator_backend"] == mod.EXACT_VALIDATOR_BACKEND
    assert artifact["exact_validator_is_oracle"] is True
    assert artifact["llm_invoked"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["n_rows"] == artifact["n_instances"] == len(rows) == 120
    assert artifact["family_counts"] == {family: 30 for family in mod.REQUIRED_FAMILIES}
    assert artifact["label_counts"] == {"invalid": 60, "valid": 60}
    assert artifact["partition_counts"] == {"dev": 24, "test": 24, "train": 72}
    assert artifact["duplicate_leakage_count"] == 0
    assert artifact["valid_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["invalid_rejection_rate"] == pytest.approx(1.0)
    assert artifact["positive_control_passed"] is True
    assert artifact["corpus_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["mutation_distance_counts"]["0"] == 60
    assert artifact["mutation_distance_counts"]["1"] > 0
    assert artifact["mutation_distance_counts"]["2"] > 0

    row_ids = {str(row["row_id"]) for row in rows}
    assert len(row_ids) == len(rows)
    for row in rows:
        assert row["schema"] == mod.CORPUS_ROW_SCHEMA
        assert row["family"] in mod.REQUIRED_FAMILIES
        assert row["partition"] in {"train", "dev", "test"}
        assert row["label"] in {"valid", "invalid"}
        assert row["exact_validator_backend"] == mod.EXACT_VALIDATOR_BACKEND
        assert row["exact_validator_is_oracle"] is True
        assert row["exact_validator_decision"] in {"accepted", "rejected"}
        assert isinstance(row["mutation_operators"], list)
        assert row["mutation_distance"] in {0, 1, 2}
        if row["label"] == "valid":
            assert row["exact_validator_decision"] == "accepted"
            assert row["mutation_distance"] == 0
        else:
            assert row["exact_validator_decision"] == "rejected"
            assert row["mutation_distance"] in {1, 2}

    mod.validate_artifact(artifact, repo_root=tmp_path)


def test_req_verify_5566_exact_controls_and_duplicate_leakage_detection() -> None:
    """REQ-VERIFY-5566: exact validators accept valid rows and reject near misses."""

    rows = mod.build_corpus_rows(_ready_upstream())
    valid = next(row for row in rows if row["label"] == "valid")
    invalid = next(row for row in rows if row["label"] == "invalid")

    assert mod.exact_validate_corpus_row(valid)["accepted"] is True
    assert mod.exact_validate_corpus_row(invalid)["accepted"] is False
    assert mod.duplicate_leakage_count(rows) == 0

    leaky = deepcopy(rows)
    duplicate = deepcopy(leaky[0])
    duplicate["row_id"] = "forced_cross_partition_duplicate"
    duplicate["partition"] = "dev" if leaky[0]["partition"] != "dev" else "test"
    leaky.append(duplicate)
    assert mod.duplicate_leakage_count(leaky) == 1

    summary = mod.summarize_rows(rows)
    assert summary["family_counts"] == {family: 30 for family in mod.REQUIRED_FAMILIES}
    assert summary["label_counts"] == {"invalid": 60, "valid": 60}
    assert summary["valid_acceptance_rate"] == pytest.approx(1.0)
    assert summary["invalid_rejection_rate"] == pytest.approx(1.0)
    assert summary["positive_control_passed"] is True


def test_req_verify_5566_validation_fails_closed_on_overclaim(tmp_path: Path) -> None:
    """REQ-VERIFY-5566: artifact validation rejects hidden LLM use and bad gates."""

    artifact = mod.run(
        repo_root=tmp_path,
        upstream_artifact=_ready_upstream(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    bad_llm = deepcopy(artifact)
    bad_llm["llm_invoked"] = True
    bad_llm["reproducibility_checksum"] = mod.payload_checksum(bad_llm)
    with pytest.raises(ValueError, match="llm_invoked"):
        mod.validate_artifact(bad_llm, repo_root=tmp_path)

    bad_oracle = deepcopy(artifact)
    bad_oracle["exact_validator_is_oracle"] = False
    bad_oracle["reproducibility_checksum"] = mod.payload_checksum(bad_oracle)
    with pytest.raises(ValueError, match="exact_validator_is_oracle"):
        mod.validate_artifact(bad_oracle, repo_root=tmp_path)

    bad_rows = deepcopy(artifact)
    bad_rows["n_rows"] = 119
    bad_rows["reproducibility_checksum"] = mod.payload_checksum(bad_rows)
    with pytest.raises(ValueError, match="n_rows"):
        mod.validate_artifact(bad_rows, repo_root=tmp_path)

    bad_controls = deepcopy(artifact)
    bad_controls["invalid_rejection_rate"] = 0.5
    bad_controls["reproducibility_checksum"] = mod.payload_checksum(bad_controls)
    with pytest.raises(ValueError, match="invalid_rejection_rate"):
        mod.validate_artifact(bad_controls, repo_root=tmp_path)

    bad_leak = deepcopy(artifact)
    bad_leak["duplicate_leakage_count"] = 1
    bad_leak["reproducibility_checksum"] = mod.payload_checksum(bad_leak)
    with pytest.raises(ValueError, match="duplicate_leakage_count"):
        mod.validate_artifact(bad_leak, repo_root=tmp_path)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("n_rows")
    missing_principle["reproducibility_checksum"] = mod.payload_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle, repo_root=tmp_path)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum, repo_root=tmp_path)


def test_req_verify_5566_source_and_corpus_errors_are_blocking(tmp_path: Path) -> None:
    """REQ-VERIFY-5566: missing source or mismatched corpus bytes fail closed."""

    assert mod.build_corpus_rows({"load_error": "missing"}) == []
    assert mod.corpus_bytes([]) == b""
    missing = mod.build_artifact(repo_root=tmp_path, upstream_artifact={"load_error": "missing"})
    assert missing["corpus_ready"] is False
    assert missing["honest_verdict"].startswith("blocked:")

    artifact = mod.run(repo_root=tmp_path, upstream_artifact=_ready_upstream())
    corpus_path = tmp_path / mod.CORPUS_RELATIVE_PATH
    corpus_path.write_text(corpus_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus_sha256"):
        mod.validate_artifact(artifact, repo_root=tmp_path)

    source_path = tmp_path / mod.SOURCE_FIXTURE_PATH
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(_ready_upstream()), encoding="utf-8")
    assert mod.load_source_artifact(tmp_path)["exact_fsm_fixture_extended_ready"] is True

    assert mod._load_json(tmp_path / "missing.json")["load_error"] == "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._load_json(malformed)["load_error"] == "json_decode"
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert mod._load_json(list_payload)["load_error"] == "json_not_object"

    with pytest.raises(ValueError, match="unknown_family"):
        mod.valid_candidate_for_family("unknown", 0)
    with pytest.raises(ValueError, match="unknown_family"):
        mod.invalid_candidate_for_family("unknown", {"candidate_kind": "asp_row", "candidate": {}}, 0, 1)
    with pytest.raises(ValueError, match="unknown_candidate_kind"):
        mod.exact_signature("unknown", {})

    summary = deepcopy(mod.summarize_rows(mod.build_corpus_rows(_ready_upstream())))
    summary["duplicate_leakage_count"] = 1
    assert "duplicate_leakage_count" in mod.readiness_blockers(True, summary)
