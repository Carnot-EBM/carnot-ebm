"""Tests for the independent contrast feature leakage audit.

Spec refs: REQ-VERIFY-6987 and SCENARIO-VERIFY-6987-*.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_6987_contrast_feature_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _feature(candidate_id: str, model_id: str, value: float = 0.25) -> dict:
    candidate_hash = exp.sha256_text(candidate_id)
    raw = [
        {
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash,
            "model_id": model_id,
            "candidate_token_position": 0,
            "sequence_position": 2,
            "selected_token_id": 7,
            "surprisal": value,
            "token_entropy": value + 0.1,
            "top_probability_margin": value + 0.2,
            "local_surprisal_change": 0.0,
        }
    ]
    return {
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "model_id": model_id,
        "terminal": True,
        "candidate_token_count": 1,
        "full_token_count": 3,
        "sequence_nll": value,
        "length_normalized_nll": value,
        "mean_token_entropy": value + 0.1,
        "mean_top_probability_margin": value + 0.2,
        "mean_local_surprisal_change": 0.0,
        "max_abs_local_surprisal_change": 0.0,
        "raw_token_count": 1,
        "raw_token_hash": exp.sha256_json(raw),
    }


def _small_bank() -> dict:
    candidate_ids = ("candidate-a", "candidate-b", "candidate-c", "candidate-d")
    labels = ("equivalent", "non_equivalent", "equivalent", "non_equivalent")
    manifest = [
        {
            "ordinal": index,
            "candidate_id": candidate_id,
            "candidate_hash": exp.sha256_text(candidate_id),
            "candidate_text": candidate_id,
            "prompt_text": "frozen prompt",
            "source_block": "exp6984",
        }
        for index, candidate_id in enumerate(candidate_ids)
    ]
    features = [
        _feature(candidate_id, model_id, 0.2 + index / 100)
        for index, candidate_id in enumerate(candidate_ids)
        for model_id in exp.REQUIRED_MODEL_IDS
    ]
    raw_rows = []
    parser_rows = []
    for feature in features:
        raw_rows.append(
            {
                "candidate_id": feature["candidate_id"],
                "candidate_hash": feature["candidate_hash"],
                "model_id": feature["model_id"],
                "candidate_token_position": 0,
                "sequence_position": 2,
                "selected_token_id": 7,
                "surprisal": feature["sequence_nll"],
                "token_entropy": feature["mean_token_entropy"],
                "top_probability_margin": feature["mean_top_probability_margin"],
                "local_surprisal_change": 0.0,
            }
        )
        parser_rows.append(
            {
                "candidate_id": feature["candidate_id"],
                "model_id": feature["model_id"],
                "json_parseable": False,
                "byte_count": len(feature["candidate_id"]),
                "line_count": 1,
                "brace_count": 0,
                "bracket_count": 0,
                "object_count": 0,
                "array_count": 0,
                "key_count": 0,
                "array_item_count": 0,
                "null_count": 0,
                "boolean_count": 0,
                "number_count": 0,
                "string_count": 0,
            }
        )
    joined = [
        {
            "candidate_id": candidate_id,
            "candidate_hash": exp.sha256_text(candidate_id),
            "model_id": model_id,
            "exact_label": label,
            "source_block": "exp6984",
        }
        for candidate_id, label in zip(candidate_ids, labels, strict=True)
        for model_id in exp.REQUIRED_MODEL_IDS
    ]
    blocks = [
        [
            {
                "candidate_id": row["candidate_id"],
                "candidate_text": row["candidate_text"],
                "prompt_text": row["prompt_text"],
            }
            for row in manifest
        ],
        [],
        [],
    ]
    return {
        "three_family_feature_bank_complete_score": 1,
        "observed_feature_row_count": len(features),
        "per_candidate_model_rows": features,
        "sequence_feature_rows": deepcopy(features),
        "raw_token_rows": raw_rows,
        "parser_feature_rows": parser_rows,
        "joined_label_rows": joined,
        "scoring_manifest_rows": manifest,
        "process_isolation_rows": [
            {
                "model_id": model_id,
                "scorer_input_field_names": sorted(exp.MODEL_INPUT_FIELDS),
                "input_payload_hash": exp.sha256_json(blocks),
                "denied_fields_visible": [],
                "passed": True,
            }
            for model_id in exp.REQUIRED_MODEL_IDS
        ],
        "label_denial_rows": [
            {"mutation_field": field, "passed": True}
            for field in sorted(exp.PROHIBITED_DIRECT_FIELDS)
        ],
    }


def _metadata() -> dict[str, dict]:
    return {
        "candidate-a": {
            "source_block": "exp6984",
            "source_pair_id": "pair-1",
            "source_group_id": "group-1",
            "split": "train",
            "pair_position": 0,
            "fault_family": "bound_change",
            "formulation_family": "linear",
            "event_ordinal": None,
        },
        "candidate-b": {
            "source_block": "exp6984",
            "source_pair_id": "pair-1",
            "source_group_id": "group-1",
            "split": "train",
            "pair_position": 1,
            "fault_family": "bound_change",
            "formulation_family": "linear",
            "event_ordinal": None,
        },
        "candidate-c": {
            "source_block": "exp6984",
            "source_pair_id": "pair-2",
            "source_group_id": "group-2",
            "split": "calibration",
            "pair_position": 0,
            "fault_family": "coefficient_swap",
            "formulation_family": "linear",
            "event_ordinal": None,
        },
        "candidate-d": {
            "source_block": "exp6984",
            "source_pair_id": "pair-2",
            "source_group_id": "group-2",
            "split": "calibration",
            "pair_position": 1,
            "fault_family": "coefficient_swap",
            "formulation_family": "linear",
            "event_ordinal": None,
        },
    }


def test_req_verify_6987_spec_precedes_implementation() -> None:
    """REQ-VERIFY-6987 defines the audit and every required evidence surface."""
    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-VERIFY-6987", 1)[1]
    for marker in (
        "SCENARIO-VERIFY-6987-PRECONDITIONS",
        "SCENARIO-VERIFY-6987-JOINS",
        "SCENARIO-VERIFY-6987-BALANCE",
        "SCENARIO-VERIFY-6987-SOURCES",
        "SCENARIO-VERIFY-6987-FUTURE",
        "SCENARIO-VERIFY-6987-LABEL-DENIAL",
        "SCENARIO-VERIFY-6987-SHORTCUTS",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_6987_hashes_precede_json_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6987-PRECONDITIONS hashes every source before JSON parsing."""
    source = tmp_path / "source.json"
    source.write_text('{"aggregate_claim": 1}\n', encoding="utf-8")

    def forbidden_parse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("changed bytes were parsed")

    monkeypatch.setattr(exp.json, "loads", forbidden_parse)
    loaded = exp.load_hashed_sources(
        tmp_path,
        source_paths={"source": Path("source.json")},
        expected_hashes={"source": "sha256:wrong"},
    )

    assert loaded["passed"] is False
    assert loaded["sources"] == {}
    assert loaded["hash_rows"][0]["check"] == "source_hash:source"


def test_scenario_verify_6987_pinned_sha256_values_are_complete() -> None:
    """SCENARIO-VERIFY-6987-PRECONDITIONS rejects truncated pinned digests."""
    for digest in (*exp.EXPECTED_SOURCE_HASHES.values(), *exp.EXPECTED_CONTENT_HASHES.values()):
        algorithm, hexadecimal = digest.split(":", 1)
        assert algorithm == "sha256"
        assert len(hexadecimal) == 64
        assert set(hexadecimal) <= set("0123456789abcdef")


def test_scenario_verify_6987_preregistered_seed_fits_probe_backend() -> None:
    """SCENARIO-VERIFY-6987-SHORTCUTS uses a valid deterministic sklearn seed."""
    assert 0 <= exp.RANDOM_SEED < 2**32


def test_scenario_verify_6987_fresh_child_writes_validated_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6987-READONLY hands the child artifact to the controller."""
    output = tmp_path / "child-result.json"
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=0.1,
        preconditions_checked=[exp.gate_check("fixture", True, False)],
        evidence={},
    )
    monkeypatch.setattr(exp, "build_from_repo", lambda *_args, **_kwargs: artifact)

    assert exp.main(["--fresh-child", "--date", "20260904", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_6987_full_replay_is_complete_and_disqualified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-6987 replays all frozen evidence without writing its sources."""
    monkeypatch.setattr(
        exp,
        "sandbox_runtime_receipt",
        lambda *_args, **_kwargs: {"passed": True, "output_is_temporary": True},
    )

    artifact = exp.build_from_repo(
        REPO_ROOT,
        run_date="20260904",
        output_path=tmp_path / "unused-child-result.json",
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["feature_audit_complete_score"] == 1
    assert artifact["contrast_feature_bank_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert len(artifact["per_candidate_model_rows"]) == 414
    failed_probes = [
        row for row in artifact["shortcut_interval_rows"] if row["gate_passed"] is False
    ]
    assert [row["probe_name"] for row in failed_probes] == ["mutation_metadata_only"]


def test_scenario_verify_6987_missing_model_and_raw_hash_block() -> None:
    """SCENARIO-VERIFY-6987-PRECONDITIONS rejects incomplete or changed raw evidence."""
    bank = _small_bank()
    expected = {
        "raw_token_rows": exp.sha256_json(bank["raw_token_rows"]),
        "joined_label_rows": exp.sha256_json(bank["joined_label_rows"]),
    }
    clean = exp.audit_bank_preconditions(
        bank,
        expected_content_hashes=expected,
        expected_feature_rows=12,
        expected_candidate_rows=4,
    )
    assert clean["passed"] is True

    missing = deepcopy(bank)
    missing["per_candidate_model_rows"].pop()
    failed = exp.audit_bank_preconditions(
        missing,
        expected_content_hashes=expected,
        expected_feature_rows=12,
        expected_candidate_rows=4,
    )
    assert failed["passed"] is False
    assert "feature_row_count" in failed["failed_checks"]

    changed = deepcopy(bank)
    changed["raw_token_rows"][0]["surprisal"] = 9.0
    failed = exp.audit_bank_preconditions(
        changed,
        expected_content_hashes=expected,
        expected_feature_rows=12,
        expected_candidate_rows=4,
    )
    assert "raw_token_rows_hash" in failed["failed_checks"]


def test_scenario_verify_6987_rebuilds_unique_three_family_joins() -> None:
    """SCENARIO-VERIFY-6987-JOINS preserves missing and duplicate join failures."""
    bank = _small_bank()
    clean = exp.rebuild_candidate_model_joins(bank, _metadata())

    assert len(clean["per_candidate_model_rows"]) == 12
    assert len(clean["join_replay_rows"]) == 12
    assert clean["source_disagreement_rows"] == []
    assert all(row["passed"] for row in clean["join_replay_rows"])
    assert all(row["complete"] for row in clean["family_coverage_rows"])

    duplicate = deepcopy(bank)
    duplicate["joined_label_rows"].append(deepcopy(duplicate["joined_label_rows"][0]))
    replay = exp.rebuild_candidate_model_joins(duplicate, _metadata())
    assert any(row["kind"] == "duplicate_label_key" for row in replay["source_disagreement_rows"])

    missing = deepcopy(bank)
    missing["joined_label_rows"].pop()
    replay = exp.rebuild_candidate_model_joins(missing, _metadata())
    assert any(row["kind"] == "missing_label_key" for row in replay["source_disagreement_rows"])


def test_scenario_verify_6987_balance_and_source_overlap_fail_closed() -> None:
    """SCENARIO-VERIFY-6987-BALANCE and SOURCES expose imbalance and overlap."""
    bank = _small_bank()
    rows = exp.rebuild_candidate_model_joins(bank, _metadata())["per_candidate_model_rows"]
    balance = exp.audit_label_balance(rows, required_splits=("train", "calibration"))
    assert all(row["exactly_balanced"] for row in balance)

    imbalanced = deepcopy(rows)
    for row in imbalanced:
        if row["candidate_id"] == "candidate-d":
            row["exact_label"] = "equivalent"
    balance = exp.audit_label_balance(imbalanced, required_splits=("train", "calibration"))
    assert any(not row["exactly_balanced"] for row in balance)

    leaked = deepcopy(rows)
    for row in leaked:
        if row["candidate_id"] in {"candidate-c", "candidate-d"}:
            row["source_group_id"] = "group-1"
    isolation = exp.audit_source_isolation(leaked)
    assert any(row["overlap_count"] == 1 for row in isolation["source_overlap_rows"])
    assert isolation["all_disjoint"] is False


def test_scenario_verify_6987_future_metadata_is_rejected() -> None:
    """SCENARIO-VERIFY-6987-FUTURE rejects later events and future fields."""
    clean = [
        {
            "candidate_id": "now",
            "event_ordinal": 4,
            "visible_event_ordinals": [0, 1, 2, 3],
            "feature_payload": {"length_normalized_nll": 1.0},
        }
    ]
    assert exp.audit_future_isolation(clean)["passed"] is True

    later = deepcopy(clean)
    later[0]["visible_event_ordinals"].append(5)
    later[0]["feature_payload"]["held_future_window"] = [6]
    result = exp.audit_future_isolation(later)
    assert result["passed"] is False
    assert result["rows"][0]["later_event_references"] == [5]
    assert result["rows"][0]["prohibited_paths"]


def test_scenario_verify_6987_direct_fields_and_training_ids_are_denied() -> None:
    """SCENARIO-VERIFY-6987-LABEL-DENIAL and SCHEMA exclude oracle and ID fields."""
    bank = _small_bank()
    denial = exp.audit_label_denial(bank, exp.TRAINING_FEATURE_FIELDS)
    assert denial["direct_leakage_count"] == 0
    assert all(row["passed"] for row in denial["label_denial_replay_rows"])

    leaked = deepcopy(bank)
    leaked["scoring_manifest_rows"][0]["candidate_text"] = json.dumps(
        {"nested": {"exact_label": "equivalent"}}
    )
    denial = exp.audit_label_denial(leaked, (*exp.TRAINING_FEATURE_FIELDS, "source_group_id"))
    assert denial["direct_leakage_count"] == 2
    paths = {row["path"] for row in denial["prohibited_field_rows"]}
    assert any("exact_label" in path for path in paths)
    assert "training_schema.source_group_id" in paths

    schema = exp.build_feature_schema_rows(
        set(bank["per_candidate_model_rows"][0]),
        set(bank["parser_feature_rows"][0]),
    )
    included = {row["field"] for row in schema if row["included_in_training_schema"]}
    assert included == set(exp.TRAINING_FEATURE_FIELDS)
    assert included.isdisjoint(exp.PROHIBITED_TRAINING_FIELDS)


def test_scenario_verify_6987_pair_grouped_provenance_probe_disqualifies() -> None:
    """SCENARIO-VERIFY-6987-SHORTCUTS catches a source-only label shortcut."""
    rows = []
    for pair_index in range(24):
        label = pair_index % 2
        source = "positive_source" if label else "negative_source"
        for candidate_index in range(2):
            candidate_id = f"c-{pair_index}-{candidate_index}"
            for model_id in exp.REQUIRED_MODEL_IDS:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model_id": model_id,
                        "source_pair_id": f"pair-{pair_index}",
                        "exact_label": "equivalent" if label else "non_equivalent",
                        "source_block": source,
                    }
                )
    result = exp.fit_shortcut_probe(
        rows,
        probe_name="source_metadata_only",
        feature_fields=("source_block",),
        random_seed=17,
        folds=4,
        bootstrap_samples=300,
    )

    assert result["interval_row"]["shortcut_auroc"] == pytest.approx(1.0)
    assert result["interval_row"]["ci95_lower"] > 0.99
    assert result["interval_row"]["gate_passed"] is False
    assert all(row["group_overlap_count"] == 0 for row in result["probe_rows"])


def test_scenario_verify_6987_readiness_is_bare_and_disqualification_terminal() -> None:
    """SCENARIO-VERIFY-6987-BARE separates audit completion from release."""
    intervals = [
        {"probe_name": "source_metadata_only", "ci95_upper": 0.91, "gate_passed": False}
    ]
    reduced = exp.reduce_readiness(
        audit_complete=True,
        direct_leakage_count=0,
        models_complete=True,
        splits_balanced=True,
        sources_disjoint=True,
        future_isolated=True,
        shortcut_interval_rows=intervals,
    )
    assert reduced == {
        "feature_audit_complete_score": 1,
        "contrast_feature_bank_ready_score": 0,
        "verdict_class": "disqualified",
        "honest_verdict": "complete_disqualified_contrast_feature_bank_shortcut_gate",
    }

    ready = exp.reduce_readiness(
        audit_complete=True,
        direct_leakage_count=0,
        models_complete=True,
        splits_balanced=True,
        sources_disjoint=True,
        future_isolated=True,
        shortcut_interval_rows=[
            {"probe_name": "model_identity_only", "ci95_upper": 0.7, "gate_passed": True}
        ],
    )
    assert type(ready["feature_audit_complete_score"]) is int
    assert type(ready["contrast_feature_bank_ready_score"]) is int
    assert ready["verdict_class"] == "circular_positive"


def test_scenario_verify_6987_artifact_schema_and_checksum_are_strict() -> None:
    """REQ-VERIFY-6987 validates principles, bare gates, verdicts, and checksums."""
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=0.5,
        preconditions_checked=[exp.gate_check("sources", True, True)],
        evidence={
            "feature_audit_complete_score": 1,
            "contrast_feature_bank_ready_score": 0,
            "direct_leakage_count": 0,
            "shortcut_interval_rows": [
                {"probe_name": "order_only", "ci95_upper": 0.99, "gate_passed": False}
            ],
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_contrast_feature_bank_shortcut_gate",
        },
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.validate_artifact(artifact) == []

    wrapped = deepcopy(artifact)
    wrapped["contrast_feature_bank_ready_score"] = {"value": 0}
    assert "contrast_feature_bank_ready_score_not_bare_int" in exp.validate_artifact(wrapped)
    changed = deepcopy(artifact)
    changed["direct_leakage_count"] = 1
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_scenario_verify_6987_blocked_artifact_names_expected_and_observed() -> None:
    """SCENARIO-VERIFY-6987-PRECONDITIONS keeps the first blocked cause structured."""
    artifact = exp.build_artifact(
        run_date="20260904",
        duration_s=0.1,
        preconditions_checked=[exp.gate_check("raw_token_rows_hash", "sha256:a", "sha256:b")],
        evidence={},
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_contrast_feature_audit"
    assert artifact["gate_check_summary"]["failed_check"] == "raw_token_rows_hash"
    assert artifact["gate_check_summary"]["expected_value"] == "sha256:a"
    assert artifact["gate_check_summary"]["observed_value"] == "sha256:b"
    assert artifact["feature_audit_complete_score"] == 0
    assert artifact["contrast_feature_bank_ready_score"] == 0


def test_scenario_verify_6987_controller_uses_linux_read_only_sandbox() -> None:
    """SCENARIO-VERIFY-6987-READONLY uses a new network namespace and hides devices."""
    command = exp.fresh_process_command(
        executable=Path("/venv/python"),
        wrapper=Path("/repo/scripts/experiment.py"),
        repo_root=Path("/repo"),
        writable_root=Path("/tmp/audit"),
        output_path=Path("/tmp/audit/result.json"),
        run_date="20260904",
    )
    joined = " ".join(str(part) for part in command)
    assert command[0] == "bwrap"
    assert "--unshare-net" in command
    assert "--unshare-pid" in command
    assert "--ro-bind / /" in joined
    assert "--dev /dev" in joined
    assert "CUDA_VISIBLE_DEVICES " in joined
    assert "--fresh-child" in command

    source = inspect.getsource(exp)
    assert "experiment_6986_three_family_contrast_features import" not in source
    wrapper = (REPO_ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_6987_contrast_feature_audit import main" in wrapper
    assert "raise SystemExit(main())" in wrapper
