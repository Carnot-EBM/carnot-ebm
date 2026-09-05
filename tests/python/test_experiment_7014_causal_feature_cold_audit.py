"""Tests for REQ-VERIFY-7014 and SCENARIO-VERIFY-7014-*.

The tests use small synthetic surfaces for attacks. The end-to-end test uses
the frozen Exp7012 and Exp7013 artifacts that the release decision audits.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_7014_causal_feature_cold_audit as exp7014
from carnot.experiment_7014_causal_feature_cold_audit import (
    ALLOWED_TENSOR_FIELDS,
    FIELD_PRINCIPLES,
    INFERENCE_SUBSTRATE,
    MODEL_FAMILIES,
    REQUIRED_ARTIFACT_FIELDS,
    AuditInputError,
    audit_identifiability,
    audit_sidecar_interventions,
    audit_tensor_payload,
    build_artifact,
    build_from_repo,
    build_learner_tensor,
    fit_grouped_probe,
    fresh_process_command,
    gate_check,
    gate_summary,
    load_hashed_sources,
    reduce_release,
    sandbox_runtime_receipt,
    sha256_path,
    validate_artifact,
    write_json_atomic,
)


def _signed_rows(pair_count: int = 12) -> list[dict]:
    """Build a complete balanced response surface with three model cells."""

    rows = []
    for pair_index in range(pair_count):
        direction = "repair" if pair_index % 2 else "violation"
        source = f"source-{(pair_index // 2) % 4}"
        for model_index, model in enumerate(MODEL_FAMILIES):
            effect = 0.40 + 0.01 * pair_index + 0.02 * model_index
            rows.append(
                {
                    "pair_id": f"pair-{pair_index:02d}",
                    "model_repository": model,
                    "comparison": f"clean_to_{direction}",
                    "signed_primary_delta": effect,
                    "signed_isomorphic_delta": effect + 0.02,
                    "isomorphic_clean_delta": 0.01,
                    "isomorphic_intervention_delta": -0.01,
                    "isomorphic_signed_control_delta": 0.02,
                    "evaluation_partition": ("train", "held_source")[pair_index % 2],
                    "held_source_family": source,
                    "terminal": True,
                }
            )
    return rows


def _sidecar(rows: list[dict]) -> list[dict]:
    """Build audit-only metadata for each response pair."""

    output = []
    for row in rows[:: len(MODEL_FAMILIES)]:
        direction = row["comparison"].removeprefix("clean_to_")
        output.append(
            {
                "block_id": row["pair_id"],
                "intervention_direction": direction,
                "mutation_kind": f"mutation-{len(output) % 4}",
                "source_family": row["held_source_family"],
                "split": row["evaluation_partition"],
                "serialization_template": "balanced",
                "prompt_char_count": 100 + 2 * len(output),
                "prompt_word_count": 20 + len(output) % 3,
                "variable_name": f"x{len(output)}",
            }
        )
    return output


@pytest.fixture(scope="module")
def frozen_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build the real audit once for artifact and mutation assertions."""

    output_path = tmp_path_factory.mktemp("exp7014") / "artifact.json"
    return build_from_repo(
        Path(__file__).resolve().parents[2],
        run_date="20260905",
        output_path=output_path,
        bootstrap_draws=64,
        runtime_receipt={
            "fresh_process": True,
            "network_namespace_isolated": True,
            "network_disabled": True,
            "gpu_devices_visible": [],
            "online_model_access": False,
            "training_disabled": True,
            "source_tree_read_only": True,
            "passed": True,
        },
    )


@pytest.mark.parametrize(
    ("payload", "expected_field"),
    [
        ({"label": 1}, "label"),
        ({"ground_truth": 1}, "ground_truth"),
        ({"targetLabel": 1}, "targetLabel"),
        ({"nested": {"provenance": "authority"}}, "nested"),
        ({"pair_id": "row-key"}, "pair_id"),
        ({"serialized_form": "{}"}, "serialized_form"),
        ({"prompt_char_count": 10}, "prompt_char_count"),
        ({"mutationKind": "repair"}, "mutationKind"),
        ({"source_family": "source"}, "source_family"),
        ({"evaluation_partition": "train"}, "evaluation_partition"),
        ({"model_repository": "model"}, "model_repository"),
        ({"score_norm": 0.5}, "score_norm"),
        ({"absolute_magnitude": 0.5}, "absolute_magnitude"),
        ({"response_hash": "sha256:bad"}, "response_hash"),
    ],
)
def test_req_7014_denies_direct_alias_nested_and_nuisance_fields(
    payload: dict, expected_field: str
) -> None:
    """SCENARIO-VERIFY-7014-LEAKAGE rejects every named substitute."""

    base = {field: 0.25 for field in ALLOWED_TENSOR_FIELDS}
    result = audit_tensor_payload([{**base, **payload}])
    assert result["direct_leakage_count"] >= 1
    assert any(row["field"] == expected_field for row in result["direct_leakage_rows"])


def test_req_7014_accepts_only_signed_response_values() -> None:
    """REQ-VERIFY-7014 allows two signed response values and nothing else."""

    result = audit_tensor_payload([{"signed_primary_delta": 0.4, "signed_isomorphic_delta": -0.2}])
    assert result == {"direct_leakage_count": 0, "direct_leakage_rows": []}


def test_req_7014_rejects_missing_and_non_numeric_signed_values() -> None:
    """SCENARIO-VERIFY-7014-LEAKAGE rejects malformed numeric payloads."""

    malformed = audit_tensor_payload([{"signed_primary_delta": "0.4"}])
    assert malformed["direct_leakage_count"] == 2
    assert {row["reason"] for row in malformed["direct_leakage_rows"]} == {
        "non_finite_or_non_numeric",
        "missing_registered_field",
    }


def test_req_7014_builds_complete_tensor_and_rejects_bad_cells() -> None:
    """SCENARIO-VERIFY-7014-PRECONDITIONS rejects incomplete family cells."""

    rows = _signed_rows()
    batch = build_learner_tensor(rows)
    assert batch.matrix.shape == (12, 6)
    assert batch.matrix.dtype == np.dtype("<f8")
    assert batch.targets == tuple(index % 2 for index in range(12))
    assert set(batch.matrix.tobytes()).issubset(set(range(256)))

    with pytest.raises(AuditInputError, match="incomplete_family_cell"):
        build_learner_tensor(rows[:-1])
    with pytest.raises(AuditInputError, match="duplicate_family_cell"):
        build_learner_tensor([*rows, deepcopy(rows[0])])
    changed = deepcopy(rows)
    changed[0]["signed_primary_delta"] = math.nan
    with pytest.raises(AuditInputError, match="non_finite_signed_response"):
        build_learner_tensor(changed)

    changed = deepcopy(rows)
    changed[0]["model_repository"] = "unknown-model"
    with pytest.raises(AuditInputError, match="incomplete_family_cell"):
        build_learner_tensor(changed)
    changed = deepcopy(rows)
    changed[0]["comparison"] = "clean_to_unknown"
    with pytest.raises(AuditInputError, match="inconsistent_intervention_direction"):
        build_learner_tensor(changed)
    changed = deepcopy(rows)
    changed[0]["held_source_family"] = "different-source"
    with pytest.raises(AuditInputError, match="inconsistent_held_source"):
        build_learner_tensor(changed)


def test_req_7014_sidecar_mutations_leave_tensor_and_prediction_identical() -> None:
    """SCENARIO-VERIFY-7014-SIDECARS covers all five sidecar attacks."""

    rows = _signed_rows()
    result = audit_sidecar_interventions(rows, _sidecar(rows), seed=7014)
    assert [row["condition"] for row in result] == [
        "correct",
        "permuted",
        "replaced",
        "deleted",
        "alpha_renamed",
    ]
    assert all(row["passed"] and not row["sidecar_open_attempted"] for row in result)
    assert len({row["tensor_hash"] for row in result}) == 1
    assert len({row["reference_prediction_hash"] for row in result}) == 1


def test_req_7014_grouped_probe_never_splits_blocks_or_bootstraps_rows() -> None:
    """SCENARIO-VERIFY-7014-PROBES keeps blocks and sources whole."""

    rows = []
    for index in range(24):
        label = index % 2
        for repeat in range(3):
            rows.append(
                {
                    "block_id": f"block-{index:02d}",
                    "source_family": f"source-{(index // 2) % 4}",
                    "target": label,
                    "balanced_feature": float((index // 2 + repeat) % 3),
                }
            )
    result = fit_grouped_probe(
        "metadata",
        rows,
        ("balanced_feature",),
        seed=7014,
        folds=4,
        bootstrap_draws=32,
    )
    assert result["interval_row"]["bootstrap_unit"] == "source_block_and_source_family"
    assert result["interval_row"]["bootstrap_samples"] == 32
    assert len(result["prediction_rows"]) == len(rows)
    assert all(
        not (set(row["train_block_ids"]) & set(row["test_block_ids"]))
        for row in result["probe_rows"]
    )
    assert all(row["sampled_row_ids"] is None for row in result["bootstrap_rows"])
    assert all(row["terminal"] for row in result["bootstrap_rows"])


def test_req_7014_probe_preconditions_fail_closed() -> None:
    """SCENARIO-VERIFY-7014-PROBES rejects invalid class and source folds."""

    with pytest.raises(AuditInputError, match="probe_requires_two_classes"):
        fit_grouped_probe("bad", [], ("x",), seed=1)
    one_source = [
        {"block_id": f"b{index}", "source_family": "one", "target": index % 2, "x": index}
        for index in range(4)
    ]
    with pytest.raises(AuditInputError, match="probe_requires_two_sources"):
        fit_grouped_probe("bad", one_source, ("x",), seed=1)
    separated_classes = [
        {"block_id": f"b{index}", "source_family": f"s{index % 2}", "target": index % 2, "x": index}
        for index in range(8)
    ]
    with pytest.raises(AuditInputError, match="source_fold_requires_two_classes"):
        fit_grouped_probe("bad", separated_classes, ("x",), seed=1, folds=2)


def test_req_7014_identifiability_is_separate_by_model_source_and_isomorph() -> None:
    """SCENARIO-VERIFY-7014-IDENTIFIABILITY keeps every direction separate."""

    rows = _signed_rows(24)
    result = audit_identifiability(rows, seed=7014, bootstrap_draws=64, tolerance=0.20)
    assert len(result["family_identifiability_rows"]) == len(MODEL_FAMILIES)
    assert len(result["held_source_identifiability_rows"]) == 24
    assert result["identifiable_family_count"] == 3
    assert all(row["passed"] for row in result["isomorphic_invariance_rows"])

    reversed_rows = deepcopy(rows)
    for row in reversed_rows:
        if row["model_repository"] == MODEL_FAMILIES[0] and row["comparison"] == "clean_to_repair":
            row["signed_isomorphic_delta"] *= -1
    failed = audit_identifiability(reversed_rows, seed=7014, bootstrap_draws=32, tolerance=0.20)
    assert not all(row["passed"] for row in failed["isomorphic_invariance_rows"])


def test_req_7014_identifiability_rejects_missing_direction_and_source_cells() -> None:
    """SCENARIO-VERIFY-7014-IDENTIFIABILITY fails on missing planned cells."""

    rows = _signed_rows(24)
    no_family_direction = [
        row
        for row in rows
        if not (
            row["model_repository"] == MODEL_FAMILIES[0] and row["comparison"] == "clean_to_repair"
        )
    ]
    with pytest.raises(AuditInputError, match="identifiability_cell_missing"):
        audit_identifiability(no_family_direction, seed=1, bootstrap_draws=2)
    missing_held_cell = [
        row
        for row in rows
        if not (
            row["model_repository"] == MODEL_FAMILIES[0]
            and row["comparison"] == "clean_to_repair"
            and row["held_source_family"] == "source-0"
        )
    ]
    with pytest.raises(AuditInputError, match="held_source_cell_missing"):
        audit_identifiability(missing_held_cell, seed=1, bootstrap_draws=2)


@pytest.mark.parametrize(
    ("kwargs", "expected_ready", "expected_class"),
    [
        ({}, 1, "positive"),
        ({"direct_leakage_count": 1}, 0, "disqualified"),
        ({"prohibited_upper": 0.80}, 0, "disqualified"),
        ({"isomorphic_invariant": False}, 0, "disqualified"),
        ({"identifiable_family_count": 1}, 0, "null"),
        ({"audit_complete": False}, 0, "partial"),
    ],
)
def test_req_7014_release_reduction_has_one_bare_readiness_field(
    kwargs: dict, expected_ready: int, expected_class: str
) -> None:
    """SCENARIO-VERIFY-7014-RELEASE separates adverse terminal meanings."""

    inputs = {
        "audit_complete": True,
        "direct_leakage_count": 0,
        "prohibited_upper": 0.79,
        "sidecar_invariant": True,
        "isomorphic_invariant": True,
        "family_cells_complete": True,
        "identifiable_family_count": 2,
    }
    inputs.update(kwargs)
    result = reduce_release(**inputs)
    assert type(result["causal_bank_audit_complete_score"]) is int
    assert type(result["causal_feature_bank_ready_score"]) is int
    assert result["causal_feature_bank_ready_score"] == expected_ready
    assert result["verdict_class"] == expected_class


def test_req_7014_hash_precondition_fails_before_parse(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7014-PRECONDITIONS reports exact hash drift."""

    source = tmp_path / "source.json"
    source.write_text("not json", encoding="utf-8")
    result = load_hashed_sources(
        tmp_path,
        {"source": Path("source.json")},
        {"source": "sha256:" + "0" * 64},
    )
    assert not result["passed"]
    assert result["values"] == {}
    assert result["checks"][0]["check"] == "source_hash:source"
    assert result["hashes"]["source"] == sha256_path(source)


def test_req_7014_missing_hash_and_parse_failure_are_terminal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7014-PRECONDITIONS covers absent and invalid sources."""

    assert sha256_path(tmp_path / "missing.json") is None
    source = tmp_path / "source.json"
    source.write_text("not json", encoding="utf-8")
    digest = sha256_path(source)
    result = load_hashed_sources(tmp_path, {"source": Path("source.json")}, {"source": str(digest)})
    assert result["checks"][-1]["check"] == "source_parse"
    assert not result["passed"]


def test_req_7014_gate_summary_promotes_first_failure() -> None:
    """REQ-VERIFY-7014 retains each check and the first failed observation."""

    checks = [gate_check("ready", 1, 1), gate_check("hash", "expected", "observed")]
    summary = gate_summary(checks)
    assert summary["failed_check"] == "hash"
    assert summary["expected_value"] == "expected"
    assert summary["observed_value"] == "observed"


def test_req_7014_blocked_artifact_is_schema_complete() -> None:
    """SCENARIO-VERIFY-7014-ARTIFACT requires all fields even when blocked."""

    artifact = build_artifact(
        run_date="20260905",
        duration_s=0.01,
        checks=[gate_check("exp7012_ready", 1, 0)],
        evidence={},
    )
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(FIELD_PRINCIPLES)
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("blocked")
    assert artifact["gate_check_summary"]["failed_check"] == "exp7012_ready"
    assert validate_artifact(artifact) == []


def test_req_7014_validator_recomputes_aggregate_claims(frozen_artifact: dict) -> None:
    """SCENARIO-VERIFY-7014-ARTIFACT rejects forged scores and checksum."""

    artifact = build_artifact(
        run_date="20260905",
        duration_s=0.01,
        checks=[gate_check("preflight", True, False)],
        evidence={},
    )
    assert validate_artifact(artifact) == []
    forged = deepcopy(artifact)
    forged["causal_feature_bank_ready_score"] = 1
    assert "checksum_mismatch" in validate_artifact(forged)
    assert "ready_verdict_mismatch" in validate_artifact(forged)

    mutations = {
        "required_fields_missing": lambda value: value.pop("rows"),
        "field_principles_missing": lambda value: value["field_principles"].pop("rows"),
        "inference_substrate_mismatch": lambda value: value.__setitem__(
            "inference_substrate", "wrong"
        ),
        "verifier_oracle_mismatch": lambda value: value.__setitem__("verifier_is_oracle", True),
        "verdict_prefix_mismatch": lambda value: value.__setitem__("honest_verdict", "wrong"),
        "bare_integer_required:direct_leakage_count": lambda value: value.__setitem__(
            "direct_leakage_count", False
        ),
        "direct_leakage_count_mismatch": lambda value: value["direct_leakage_rows"][0].__setitem__(
            "leakage_detected", True
        ),
        "prohibited_upper_mismatch": lambda value: value.__setitem__(
            "prohibited_auroc_upper_bound_max", 0.123
        ),
        "identifiable_family_count_mismatch": lambda value: value.__setitem__(
            "identifiable_family_count", 99
        ),
        "ready_verdict_mismatch": lambda value: value.__setitem__(
            "causal_feature_bank_ready_score", 1 - value["causal_feature_bank_ready_score"]
        ),
    }
    for expected, mutate in mutations.items():
        attacked = deepcopy(frozen_artifact)
        mutate(attacked)
        assert any(error.startswith(expected) for error in validate_artifact(attacked))


def test_req_7014_build_blocks_on_source_or_runtime_preconditions(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7014-PRECONDITIONS writes both blocked paths."""

    missing = build_from_repo(tmp_path, output_path=tmp_path / "missing.json")
    assert missing["honest_verdict"] == "blocked_causal_feature_audit"
    runtime = build_from_repo(
        Path(__file__).resolve().parents[2],
        output_path=tmp_path / "runtime.json",
        runtime_receipt={},
    )
    assert runtime["gate_check_summary"]["failed_check"] == "fresh_process"


def test_req_7014_probe_join_attacks_fail_closed() -> None:
    """REQ-VERIFY-7014 rejects incomplete, conflicting, and unequal probe joins."""

    rows = _signed_rows()
    sidecar = _sidecar(rows)
    conditions = []
    for row in rows:
        conditions.extend(
            {
                "pair_id": row["pair_id"],
                "model_repository": row["model_repository"],
                "prompt_char_count": 100,
                "prompt_word_count": 20,
            }
            for _ in range(4)
        )
    expanded_sidecar = [dict(row, role=role) for row in sidecar for role in range(4)]
    with pytest.raises(AuditInputError, match="probe_join_incomplete"):
        exp7014._build_probe_input_rows(rows, expanded_sidecar[:-1], conditions)
    conflicting = deepcopy(expanded_sidecar)
    conflicting[0]["mutation_kind"] = "conflict"
    with pytest.raises(AuditInputError, match="probe_join_conflict"):
        exp7014._build_probe_input_rows(rows, conflicting, conditions)
    unequal = deepcopy(conditions)
    unequal[0]["prompt_char_count"] = 101
    with pytest.raises(AuditInputError, match="probe_length_conflict"):
        exp7014._build_probe_input_rows(rows, expanded_sidecar, unequal)


def test_req_7014_sandbox_command_receipts_and_atomic_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7014-PRECONDITIONS measures sandbox and write failures."""

    monkeypatch.setenv("CARNOT_EXP7014_PARENT_PID", "999999")
    monkeypatch.setenv("CARNOT_EXP7014_PARENT_NETNS", "different")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("CARNOT_TRAINING_DISABLED", "1")
    writable = sandbox_runtime_receipt(tmp_path)
    assert not writable["source_tree_read_only"]

    original_write_text = Path.write_text

    def deny_write(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == ".exp7014-write-probe":
            raise OSError("read only")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", deny_write)
    readonly = sandbox_runtime_receipt(tmp_path)
    assert readonly["source_tree_read_only"]
    command = fresh_process_command(
        executable=Path("/python"),
        wrapper=Path("/wrapper"),
        repo_root=tmp_path,
        writable_root=tmp_path,
        output_path=tmp_path / "out.json",
        run_date="20260905",
    )
    assert command[0] == "bwrap"
    assert "--unshare-net" in command

    monkeypatch.setattr(json, "dump", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        write_json_atomic(tmp_path / "failed.json", {"x": 1})
    assert not list(tmp_path.glob(".failed.json.*"))

    original_unlink = os.unlink
    monkeypatch.setattr(os, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        write_json_atomic(tmp_path / "unlink-failed.json", {"x": 1})
    monkeypatch.setattr(os, "unlink", original_unlink)
    for temporary in tmp_path.glob(".unlink-failed.json.*"):
        temporary.unlink()


def test_req_7014_frozen_artifacts_build_terminal_audit(frozen_artifact: dict) -> None:
    """REQ-VERIFY-7014 rebuilds the real matrix without writing source files."""

    artifact = frozen_artifact
    assert artifact["causal_bank_audit_complete_score"] == 1
    assert artifact["causal_feature_bank_ready_score"] in {0, 1}
    assert artifact["verifier_is_oracle"] is False
    assert len(artifact["learner_tensor_rows"]) == 48
    assert len(artifact["per_pair_results"]) == 48
    intervals = [
        row
        for field in REQUIRED_ARTIFACT_FIELDS
        if field.endswith("_probe_rows")
        for row in artifact[field]
        if row.get("record_type") == "interval"
    ]
    assert intervals
    assert all(row["auroc"] >= 0.5 for row in intervals)
    assert validate_artifact(artifact) == []
