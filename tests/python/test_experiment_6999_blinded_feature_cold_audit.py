"""Tests for the fresh-process blinded feature audit.

Spec refs: REQ-VERIFY-6999 and SCENARIO-VERIFY-6999-*.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_6999_blinded_feature_cold_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _raw_rows(candidate_count: int = 4) -> list[dict]:
    rows = []
    for candidate_index in range(candidate_count):
        candidate_id = f"candidate-{candidate_index}"
        prompt_hash = exp.sha256_text(f"prompt-{candidate_index // 2}")
        candidate_hash = exp.sha256_text(candidate_id)
        for family_index, model_id in enumerate(exp.REQUIRED_MODEL_IDS):
            value = float(candidate_index * 10 + family_index)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "prompt_hash": prompt_hash,
                    "candidate_hash": candidate_hash,
                    "model_id": model_id,
                    **{
                        field: value + field_index / 100
                        for field_index, field in enumerate(exp.SOURCE_FEATURES)
                    },
                }
            )
    return rows


def _authority_rows(wide_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = []
    mutations = []
    for index, row in enumerate(wide_rows):
        key = row["candidate_key"]
        pair_index = index // 2
        labels.append(
            {
                "candidate_key": key,
                "exact_label": "equivalent" if index % 2 == 0 else "non_equivalent",
                "split": "train" if pair_index % 2 == 0 else "held_out",
            }
        )
        mutations.append(
            {
                "candidate_key": key,
                "source_record_id": f"record-{index}",
                "source": {
                    "source_pair_id": f"pair-{pair_index}",
                    "source_group_id": f"group-{pair_index}",
                    "source_block": "fixture",
                },
                "mutation": {"fault_family": "bound" if index % 2 else "none"},
                "witnesses": {},
                "authority_records": [],
            }
        )
    return labels, mutations


def _write_bound_jsonl(path: Path, rows: list[dict], *, kind: str) -> None:
    ordered = sorted(rows, key=lambda row: row["candidate_key"])
    payload = b"".join((exp.canonical_json(row) + "\n").encode("utf-8") for row in ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    manifest = {
        "schema": "carnot.exp6997.immutable_jsonl_manifest.v1",
        "kind": kind,
        "path": str(path),
        "row_count": len(ordered),
        "byte_count": len(payload),
        "file_hash": exp.sha256_bytes(payload),
        "ordered_key_hashes": [exp.sha256_text(str(row["candidate_key"])) for row in ordered],
        "feature_allowlist": list(exp.FEATURE_ALLOWLIST) if kind == "learner_view" else [],
        "binding_phase": "before_authority_join",
    }
    Path(str(path) + ".manifest.json").write_text(
        exp.canonical_json(manifest) + "\n", encoding="utf-8"
    )


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    rebuilt = exp.rebuild_wide_rows(_raw_rows(), expected_candidate_count=4)
    wide = rebuilt["learner_rows"]
    labels, mutations = _authority_rows(wide)
    learner = tmp_path / "learner_view.jsonl"
    label = tmp_path / "label_split_sidecar.jsonl"
    mutation = tmp_path / "mutation_authority_sidecar.jsonl"
    _write_bound_jsonl(learner, wide, kind="learner_view")
    _write_bound_jsonl(label, labels, kind="label_split")
    _write_bound_jsonl(mutation, mutations, kind="mutation_authority")
    return learner, label, mutation, wide


def _probe_rows(pair_count: int = 24) -> list[dict]:
    rows = []
    for pair_index in range(pair_count):
        for position in range(2):
            label = position
            rows.append(
                {
                    "candidate_key": f"candidate-{pair_index}-{position}",
                    "source_pair_id": f"pair-{pair_index}",
                    "label": label,
                    "signal": float(label),
                    "noise": float((pair_index + position) % 3),
                }
            )
    return rows


def test_req_verify_6999_spec_precedes_implementation() -> None:
    """REQ-VERIFY-6999 defines every cold-audit evidence surface."""
    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-VERIFY-6999", 1)[1]
    for marker in (
        "SCENARIO-VERIFY-6999-PRECONDITIONS",
        "SCENARIO-VERIFY-6999-REBUILD",
        "SCENARIO-VERIFY-6999-LEAKAGE",
        "SCENARIO-VERIFY-6999-SIDECARS",
        "SCENARIO-VERIFY-6999-SPLITS",
        "SCENARIO-VERIFY-6999-COMMITMENT",
        "SCENARIO-VERIFY-6999-SHORTCUTS",
        "SCENARIO-VERIFY-6999-BARE",
        "SCENARIO-VERIFY-6999-READONLY",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_6999_hashes_precede_source_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6999-PRECONDITIONS rejects changed bytes before parsing."""
    source = tmp_path / "source.json"
    source.write_text('{"claim": 1}\n', encoding="utf-8")

    def forbidden_parse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("changed source was parsed")

    monkeypatch.setattr(exp.json, "loads", forbidden_parse)
    loaded = exp.load_hashed_inputs(
        tmp_path,
        source_paths={"source": Path("source.json")},
        expected_hashes={"source": "sha256:wrong"},
    )

    assert loaded["passed"] is False
    assert loaded["values"] == {}
    assert loaded["checks"][0]["check"] == "source_hash:source"


def test_scenario_verify_6999_rebuild_is_order_invariant_and_complete() -> None:
    """SCENARIO-VERIFY-6999-REBUILD pivots three families into neutral positions."""
    rows = _raw_rows()
    first = exp.rebuild_wide_rows(rows, expected_candidate_count=4)
    second = exp.rebuild_wide_rows(list(reversed(rows)), expected_candidate_count=4)

    assert first["source_disagreement_rows"] == []
    assert first["learner_rows"] == second["learner_rows"]
    assert len(first["learner_rows"]) == 4
    assert len(first["raw_rebuild_rows"]) == 12
    assert all(row["passed"] for row in first["family_completeness_rows"])
    assert all(len(row["features"]) == 63 for row in first["learner_rows"])


def test_scenario_verify_6999_rebuild_preserves_duplicate_and_missing_families() -> None:
    """SCENARIO-VERIFY-6999-REBUILD retains duplicate keys and missing families."""
    rows = _raw_rows()
    duplicate = deepcopy(rows)
    duplicate.append(deepcopy(rows[0]))
    result = exp.rebuild_wide_rows(duplicate, expected_candidate_count=4)
    assert any(row["kind"] == "duplicate_family_row" for row in result["source_disagreement_rows"])

    missing = exp.rebuild_wide_rows(rows[:-1], expected_candidate_count=4)
    assert any(
        row["kind"] == "missing_model_families" for row in missing["source_disagreement_rows"]
    )
    conflict = deepcopy(rows)
    conflict[0]["prompt_hash"] = exp.sha256_text("changed")
    result = exp.rebuild_wide_rows(conflict, expected_candidate_count=4)
    assert any(
        row["kind"] == "candidate_semantic_hash_conflict"
        for row in result["source_disagreement_rows"]
    )


def test_scenario_verify_6999_direct_nested_and_alias_leakage_is_denied() -> None:
    """SCENARIO-VERIFY-6999-LEAKAGE normalizes aliases at every depth."""
    clean = exp.rebuild_wide_rows(_raw_rows(), expected_candidate_count=4)["learner_rows"]
    assert exp.audit_learner_schema(clean, exp.FEATURE_ALLOWLIST)["passed"] is True

    leaked = deepcopy(clean)
    leaked[0]["groundTruth"] = 1
    leaked[1]["payload"] = {"sourceId": "secret"}
    leaked[2]["prompt_condition_id"] = "hint"
    leaked[3]["commitmentLatency"] = 0.25
    audit = exp.audit_learner_schema(leaked, exp.FEATURE_ALLOWLIST)
    paths = {row["path"] for row in audit["prohibited_field_rows"]}

    assert audit["direct_leakage_count"] >= 4
    assert any("groundTruth" in path for path in paths)
    assert any("sourceId" in path for path in paths)
    assert any("prompt_condition_id" in path for path in paths)
    assert any("commitmentLatency" in path for path in paths)


def test_scenario_verify_6999_loader_opens_only_learner_files(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6999-SIDECARS gives the learner no sidecar parameter."""
    learner_path, _label_path, mutation_path, _wide = _bundle(tmp_path)
    batch = exp.load_learner_view(learner_path, exp.FEATURE_ALLOWLIST)

    assert list(inspect.signature(exp.load_learner_view).parameters) == [
        "learner_path",
        "allowlist",
    ]
    assert batch.matrix.shape == (4, 63)
    assert {Path(row["path"]).name for row in batch.file_open_receipts} == {
        "learner_view.jsonl",
        "learner_view.jsonl.manifest.json",
    }
    assert str(mutation_path) not in {row["path"] for row in batch.file_open_receipts}

    with pytest.raises(exp.IsolationError, match="learner_path_required"):
        exp.load_learner_view(mutation_path, exp.FEATURE_ALLOWLIST)


def test_scenario_verify_6999_loader_rejects_duplicate_keys_and_hash_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6999-LEAKAGE rejects duplicate keys before tensors exist."""
    learner_path, _label_path, _mutation_path, wide = _bundle(tmp_path)
    _write_bound_jsonl(learner_path, [wide[0], wide[0]], kind="learner_view")
    with pytest.raises(exp.IsolationError, match="learner_schema_rejected"):
        exp.load_learner_view(learner_path, exp.FEATURE_ALLOWLIST)

    _write_bound_jsonl(learner_path, wide, kind="learner_view")
    learner_path.write_bytes(learner_path.read_bytes() + b"\n")
    with pytest.raises(exp.HashMismatchError, match="file_hash_mismatch"):
        exp.load_learner_view(learner_path, exp.FEATURE_ALLOWLIST)


def test_scenario_verify_6999_all_sidecar_conditions_are_invariant(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6999-SIDECARS covers permutation replacement and removal."""
    learner_path, _label_path, mutation_path, _wide = _bundle(tmp_path)
    result = exp.audit_sidecar_conditions(
        learner_path,
        exp.FEATURE_ALLOWLIST,
        mutation_path,
        tmp_path / "conditions",
        random_seed=17,
    )

    assert [row["condition"] for row in result["sidecar_condition_rows"]] == [
        "correct",
        "permutation",
        "cross_pair_replacement",
        "empty",
        "removed",
    ]
    assert all(row["passed"] for row in result["learner_invariance_rows"])
    assert all(
        row["sidecar_open_attempted"] is False for row in result["sidecar_open_attempt_rows"]
    )
    assert len({row["tensor_hash"] for row in result["sidecar_condition_rows"]}) == 1
    assert len({row["prediction_hash"] for row in result["sidecar_condition_rows"]}) == 1
    assert result["sidecar_permutation_rows"][0]["row_order_changed"] is True
    assert result["sidecar_replacement_rows"][0]["replacement_count"] > 0
    assert result["sidecar_removal_rows"][0]["sidecar_exists"] is False


def test_scenario_verify_6999_labels_and_sources_stay_outside_tensors(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6999-SPLITS joins routing after learner materialization."""
    learner_path, label_path, mutation_path, _wide = _bundle(tmp_path)
    learner = exp.load_learner_view(learner_path, exp.FEATURE_ALLOWLIST)
    joined = exp.join_authority(learner, label_path, mutation_path)
    audit = exp.audit_splits(joined)

    assert np.array_equal(joined.matrix, learner.matrix)
    assert all(
        row["exactly_balanced"]
        for row in audit["label_balance_rows"]
        if row["candidate_count"] > 0
    )
    assert audit["all_sources_disjoint"] is True
    assert all(row["partition_count"] == 1 for row in audit["split_isolation_rows"])

    leaked = deepcopy(list(joined.rows))
    leaked[-1]["source"]["source_group_id"] = leaked[0]["source"]["source_group_id"]
    overlap = exp.audit_splits(exp.authority_batch_from_rows(learner, leaked))
    assert overlap["all_sources_disjoint"] is False


def test_scenario_verify_6999_commitment_controls_are_admitted_only_as_prohibited() -> None:
    """SCENARIO-VERIFY-6999-COMMITMENT rejects incomplete or learner-facing controls."""
    artifact = {
        "commitment_control_complete_score": 1,
        "observed_unit_count": 2,
        "expected_unit_count": 2,
        "audit_only_control": True,
        "learner_feature_allowed": False,
        "per_candidate_condition_model_rows": [
            {
                "candidate_id": "a",
                "pair_id": "pair",
                "condition": "clean",
                "model_id": "model",
                "first_commitment_latency": 0.25,
                "commitment_range": 0.5,
                "mean_uncommitted_mass": 0.2,
                "mean_uncertainty": 0.3,
                "choice_flip_count": 0,
                "terminal": True,
            },
            {
                "candidate_id": "b",
                "pair_id": "pair",
                "condition": "clean",
                "model_id": "model",
                "first_commitment_latency": 0.5,
                "commitment_range": 0.2,
                "mean_uncommitted_mass": 0.4,
                "mean_uncertainty": 0.6,
                "choice_flip_count": 1,
                "terminal": True,
            },
        ],
        "late_label_join_rows": [
            {
                "candidate_id": "a",
                "condition": "clean",
                "model_id": "model",
                "exact_label": "equivalent",
            },
            {
                "candidate_id": "b",
                "condition": "clean",
                "model_id": "model",
                "exact_label": "non_equivalent",
            },
        ],
    }
    result = exp.build_commitment_probe_rows(artifact, expected_count=2)
    assert result["passed"] is True
    assert len(result["probe_rows"]) == 2
    assert all(row["allowed_in_learner"] is False for row in result["commitment_prohibition_rows"])

    changed = deepcopy(artifact)
    changed["learner_feature_allowed"] = True
    assert exp.build_commitment_probe_rows(changed, expected_count=2)["passed"] is False
    changed = deepcopy(artifact)
    changed["per_candidate_condition_model_rows"].pop()
    assert exp.build_commitment_probe_rows(changed, expected_count=2)["passed"] is False


def test_scenario_verify_6999_grouped_probe_reports_coefficients_and_predictions() -> None:
    """SCENARIO-VERIFY-6999-SHORTCUTS keeps source pairs within folds."""
    result = exp.fit_grouped_probe(
        _probe_rows(),
        probe_name="mutation_metadata_only",
        feature_fields=("signal",),
        random_seed=19,
        folds=4,
        bootstrap_samples=200,
        prohibited=True,
    )

    assert result["interval_row"]["shortcut_auroc"] == pytest.approx(1.0)
    assert result["interval_row"]["ci95_lower"] > 0.99
    assert result["interval_row"]["gate_passed"] is False
    assert len(result["prediction_rows"]) == 48
    assert all(row["group_overlap_count"] == 0 for row in result["probe_rows"])
    assert all(row["coefficients"] for row in result["probe_rows"])


def test_scenario_verify_6999_model_identity_probe_and_threshold_boundary() -> None:
    """SCENARIO-VERIFY-6999-SHORTCUTS treats model identity as prohibited."""
    rows = _probe_rows()
    for index, row in enumerate(rows):
        row["model_id"] = "positive-model" if row["label"] else "negative-model"
    result = exp.fit_grouped_probe(
        rows,
        probe_name="model_identity_only",
        feature_fields=("model_id",),
        random_seed=23,
        folds=4,
        bootstrap_samples=200,
        prohibited=True,
    )
    assert result["interval_row"]["ci95_upper"] >= exp.SHORTCUT_AUROC_LIMIT
    assert result["interval_row"]["prohibited"] is True
    assert exp.interval_passes(0.799999) is True
    assert exp.interval_passes(0.80) is False


def test_scenario_verify_6999_readiness_is_bare_and_terminal() -> None:
    """SCENARIO-VERIFY-6999-BARE separates completed audit from release."""
    common = {
        "audit_complete": True,
        "direct_leakage_count": 0,
        "learner_invariant": True,
        "splits_balanced": True,
        "sources_disjoint": True,
        "models_complete": True,
        "prohibited_columns_absent": True,
    }
    disqualified = exp.reduce_readiness(
        **common,
        shortcut_interval_rows=[
            {"probe_name": "source_metadata_only", "prohibited": True, "ci95_upper": 0.80}
        ],
    )
    assert disqualified == {
        "feature_isolation_audit_complete_score": 1,
        "blinded_feature_bank_ready_score": 0,
        "verdict_class": "disqualified",
        "honest_verdict": "complete_disqualified_blinded_feature_shortcut_gate",
    }
    ready = exp.reduce_readiness(
        **common,
        shortcut_interval_rows=[
            {"probe_name": "source_metadata_only", "prohibited": True, "ci95_upper": 0.79},
            {"probe_name": "allowed_learner_table", "prohibited": False, "ci95_upper": 1.0},
        ],
    )
    assert type(ready["feature_isolation_audit_complete_score"]) is int
    assert type(ready["blinded_feature_bank_ready_score"]) is int
    assert ready["verdict_class"] == "circular_positive"


def test_scenario_verify_6999_artifact_schema_checksum_and_blocked_gate() -> None:
    """SCENARIO-VERIFY-6999-BARE validates principles and blocked diagnostics."""
    blocked = exp.build_artifact(
        run_date="20260905",
        duration_s=0.1,
        checks=[exp.gate_check("structured_gate:exp6997", 1, 0)],
        evidence={},
    )

    assert exp.validate_artifact(blocked) == []
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_blinded_feature_cold_audit"
    assert blocked["gate_check_summary"]["failed_check"] == "structured_gate:exp6997"
    assert set(blocked["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)

    changed = deepcopy(blocked)
    changed["blinded_feature_bank_ready_score"] = False
    changed["field_principles"].pop("rows")
    errors = exp.validate_artifact(changed)
    assert "blinded_feature_bank_ready_score_not_bare_int" in errors
    assert "field_principles_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_scenario_verify_6999_fresh_command_disables_external_capabilities(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6999-READONLY creates a restricted child command."""
    command = exp.fresh_process_command(
        executable=Path("/venv/python"),
        wrapper=Path("/repo/wrapper.py"),
        repo_root=Path("/repo"),
        writable_root=tmp_path,
        output_path=tmp_path / "result.json",
        run_date="20260905",
    )
    joined = " ".join(str(value) for value in command)
    assert "--unshare-net" in command
    assert "CUDA_VISIBLE_DEVICES  " in joined
    assert "HF_HUB_OFFLINE 1" in joined
    assert "CARNOT_TRAINING_DISABLED 1" in joined
    assert command.count("--ro-bind") >= 1


def test_req_verify_6999_full_replay_finishes_with_terminal_shortcut_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6999 rebuilds the real frozen bank without changing sources."""
    monkeypatch.setattr(
        exp,
        "sandbox_runtime_receipt",
        lambda *_args, **_kwargs: {
            "passed": True,
            "source_tree_read_only": True,
            "training_disabled": True,
        },
    )
    artifact = exp.build_from_repo(REPO_ROOT, run_date="20260905")

    assert exp.validate_artifact(artifact) == []
    assert artifact["feature_isolation_audit_complete_score"] == 1
    assert artifact["verdict_class"] in {"circular_positive", "disqualified"}
    assert len(artifact["per_candidate_rows"]) == 138
    assert len(artifact["raw_rebuild_rows"]) == 414
    assert len(artifact["shortcut_interval_rows"]) == 7
    assert {row["probe_name"] for row in artifact["shortcut_interval_rows"]} == set(
        exp.SHORTCUT_PROBE_SPECS
    )


def test_scenario_verify_6999_wrapper_and_child_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6999-READONLY validates child output in a temporary path."""
    output = tmp_path / "child.json"
    artifact = exp.build_artifact(
        run_date="20260905",
        duration_s=0.1,
        checks=[exp.gate_check("blocked", True, False)],
        evidence={},
    )
    monkeypatch.setattr(exp, "build_from_repo", lambda *_args, **_kwargs: artifact)

    assert exp.main(["--fresh-child", "--date", "20260905", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    wrapper = (REPO_ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "carnot.experiment_6999_blinded_feature_cold_audit" in wrapper
