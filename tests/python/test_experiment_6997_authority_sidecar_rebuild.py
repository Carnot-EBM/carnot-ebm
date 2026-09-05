"""Tests for the authority-sidecar feature-bank rebuild.

Spec refs: REQ-VERIFY-6997 and SCENARIO-VERIFY-6997-*.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest

from carnot import experiment_6997_authority_sidecar_rebuild as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_index in range(2):
        candidate_id = f"candidate-{candidate_index}"
        prompt_hash = exp.sha256_text(f"prompt-{candidate_index}")
        candidate_hash = exp.sha256_text(f"candidate-text-{candidate_index}")
        for family_index, model_id in enumerate(exp.REQUIRED_MODEL_IDS):
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "prompt_hash": prompt_hash,
                "candidate_hash": candidate_hash,
                "model_id": model_id,
            }
            for feature_index, feature in enumerate(exp.SOURCE_FEATURES):
                row[feature] = float(100 * candidate_index + 10 * family_index + feature_index)
            rows.append(row)
    return rows


def _authority_rows(source_rows: list[dict[str, object]]) -> tuple[list[dict], list[dict]]:
    candidate_rows = source_rows[:: len(exp.REQUIRED_MODEL_IDS)]
    labels = []
    mutations = []
    for index, row in enumerate(candidate_rows):
        key = exp.candidate_key(str(row["prompt_hash"]), str(row["candidate_hash"]))
        labels.append(
            {
                "candidate_key": key,
                "exact_label": "equivalent" if index == 0 else "non_equivalent",
                "split": "train" if index == 0 else "held_out",
            }
        )
        mutations.append(
            {
                "candidate_key": key,
                "source_record_id": f"source-{index}",
                "source": {"source_group_id": f"group-{index}"},
                "mutation": {"fault_family": None if index == 0 else "bound_change"},
                "witnesses": {"witness_id": f"witness-{index}"},
                "authority_records": [{"authority_id": f"authority-{index}", "proved": True}],
            }
        )
    return labels, mutations


def _bundle(tmp_path: Path) -> tuple[dict[str, object], exp.LearnerBatch]:
    source_rows = _source_rows()
    rebuilt = exp.build_wide_learner_view(source_rows, expected_candidate_count=2)
    labels, mutations = _authority_rows(source_rows)
    bundle = exp.write_sidecar_bundle(
        tmp_path,
        rebuilt["learner_rows"],
        labels,
        mutations,
        exp.FEATURE_ALLOWLIST,
    )
    learner = exp.load_learner_view(bundle["learner_view_path"], exp.FEATURE_ALLOWLIST)
    return bundle, learner


def test_req_verify_6997_spec_precedes_implementation() -> None:
    """REQ-VERIFY-6997 defines each isolation and evidence surface."""
    text = (REPO_ROOT / "openspec/capabilities/constraint-verification/spec.md").read_text(
        encoding="utf-8"
    )
    section = text.split("### REQ-VERIFY-6997", 1)[1]
    for marker in (
        "SCENARIO-VERIFY-6997-PRECONDITIONS",
        "SCENARIO-VERIFY-6997-KEYS",
        "SCENARIO-VERIFY-6997-FAMILIES",
        "SCENARIO-VERIFY-6997-ALLOWLIST",
        "SCENARIO-VERIFY-6997-LOADER",
        "SCENARIO-VERIFY-6997-JOIN",
        "SCENARIO-VERIFY-6997-HASH",
        "SCENARIO-VERIFY-6997-INVARIANCE",
        "SCENARIO-VERIFY-6997-BARE",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_6997_keys_ignore_nonsemantic_metadata() -> None:
    """SCENARIO-VERIFY-6997-KEYS uses only semantic prompt and candidate hashes."""
    prompt_hash = exp.sha256_text("prompt")
    candidate_hash = exp.sha256_text("candidate")
    expected = exp.candidate_key(prompt_hash, candidate_hash)
    variants = [
        {"source": "changed"},
        {"mutation": "changed"},
        {"order": 99},
        {"split": "future"},
        {"label": "changed"},
        {"candidate_id": "changed"},
        {"model_id": "changed"},
    ]

    assert all(
        exp.candidate_key(prompt_hash, candidate_hash, **metadata) == expected
        for metadata in variants
    )
    assert exp.candidate_key(prompt_hash, exp.sha256_text("other")) != expected


def test_scenario_verify_6997_duplicate_keys_and_missing_families_fail() -> None:
    """SCENARIO-VERIFY-6997-FAMILIES rejects duplicate and incomplete family rows."""
    rows = _source_rows()
    with pytest.raises(exp.RebuildError, match="duplicate_family_row"):
        exp.build_wide_learner_view([*rows, deepcopy(rows[0])], expected_candidate_count=2)

    with pytest.raises(exp.RebuildError, match="missing_model_families"):
        exp.build_wide_learner_view(rows[:-1], expected_candidate_count=2)

    conflict = deepcopy(rows)
    conflict[1]["prompt_hash"] = exp.sha256_text("conflicting-prompt")
    with pytest.raises(exp.RebuildError, match="candidate_semantic_hash_conflict"):
        exp.build_wide_learner_view(conflict, expected_candidate_count=2)


def test_scenario_verify_6997_wide_rows_have_neutral_numeric_positions() -> None:
    """REQ-VERIFY-6997 pivots three families without exposing model identity."""
    rebuilt = exp.build_wide_learner_view(_source_rows(), expected_candidate_count=2)

    assert len(rebuilt["source_model_rows"]) == 6
    assert len(rebuilt["learner_rows"]) == 2
    assert len(exp.FEATURE_ALLOWLIST) == len(exp.REQUIRED_MODEL_IDS) * len(exp.SOURCE_FEATURES)
    assert all(name.startswith("family_") for name in exp.FEATURE_ALLOWLIST)
    assert not any(
        model_id in json.dumps(rebuilt["learner_rows"]) for model_id in exp.REQUIRED_MODEL_IDS
    )
    for row in rebuilt["learner_rows"]:
        assert set(row) == {"candidate_key", "features"}
        assert len(row["features"]) == len(exp.FEATURE_ALLOWLIST)
        assert all(type(value) in (int, float) for value in row["features"])


@pytest.mark.parametrize(
    ("mutation", "receipt_name"),
    [
        ({"exact_label": "equivalent"}, "prohibited_feature_rows"),
        ({"mutationProvenance": "bound_change"}, "prohibited_feature_rows"),
        ({"metadata": {"source": "secret"}}, "nested_metadata_rows"),
        ({"model_identity": "repository/family"}, "categorical_identity_rows"),
    ],
)
def test_scenario_verify_6997_labels_aliases_nested_and_identity_are_denied(
    mutation: dict[str, object], receipt_name: str
) -> None:
    """SCENARIO-VERIFY-6997-ALLOWLIST records each denied learner-field class."""
    rebuilt = exp.build_wide_learner_view(_source_rows(), expected_candidate_count=2)
    rows = deepcopy(rebuilt["learner_rows"])
    rows[0].update(mutation)

    receipts = exp.learner_schema_receipts(rows, exp.FEATURE_ALLOWLIST)

    assert receipts[receipt_name]
    assert receipts["passed"] is False


def test_scenario_verify_6997_loader_api_and_file_receipts_deny_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6997-LOADER exposes only its learner path and allowlist."""
    bundle, learner = _bundle(tmp_path)
    signature = inspect.signature(exp.load_learner_view)

    assert list(signature.parameters) == ["learner_path", "allowlist"]
    with pytest.raises(TypeError):
        exp.load_learner_view(  # type: ignore[call-arg]
            bundle["learner_view_path"], exp.FEATURE_ALLOWLIST, object()
        )
    with pytest.raises(TypeError):
        exp.load_learner_view(  # type: ignore[call-arg]
            bundle["learner_view_path"], exp.FEATURE_ALLOWLIST, callback=lambda: None
        )
    with pytest.raises(exp.IsolationError, match="learner_path_required"):
        exp.load_learner_view(bundle["label_split_sidecar_path"], exp.FEATURE_ALLOWLIST)

    monkeypatch.setenv("LABEL_SPLIT_SIDECAR", str(bundle["label_split_sidecar_path"]))
    monkeypatch.setenv("MUTATION_AUTHORITY_SIDECAR", str(bundle["mutation_authority_sidecar_path"]))
    repeated = exp.load_learner_view(bundle["learner_view_path"], exp.FEATURE_ALLOWLIST)
    opened = {row["path"] for row in repeated.file_open_receipts}
    assert str(bundle["label_split_sidecar_path"]) not in opened
    assert str(bundle["mutation_authority_sidecar_path"]) not in opened
    assert learner.tensor_hash == repeated.tensor_hash


def test_scenario_verify_6997_manifests_bind_bytes_before_join(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6997-SIDECARS binds paths, rows, keys, bytes, and hashes."""
    bundle, _learner = _bundle(tmp_path)

    prefixes = (
        ("learner_view", "learner_view_manifest_rows"),
        ("label_split_sidecar", "label_split_manifest_rows"),
        ("mutation_authority_sidecar", "mutation_authority_manifest_rows"),
    )
    for prefix, manifest_field in prefixes:
        manifest = bundle[manifest_field][0]
        path = Path(bundle[f"{prefix}_path"])
        assert manifest["path"] == str(path)
        assert manifest["row_count"] == 2
        assert manifest["byte_count"] == path.stat().st_size
        assert manifest["file_hash"] == exp.sha256_bytes(path.read_bytes())
        assert len(manifest["ordered_key_hashes"]) == 2


def test_scenario_verify_6997_hash_mismatch_fails_before_return(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6997-HASH rejects changed learner and authority bytes."""
    bundle, learner = _bundle(tmp_path)
    learner_path = Path(bundle["learner_view_path"])
    learner_path.chmod(0o644)
    learner_path.write_bytes(learner_path.read_bytes() + b"\n")
    with pytest.raises(exp.HashMismatchError, match="file_hash_mismatch"):
        exp.load_learner_view(learner_path, exp.FEATURE_ALLOWLIST)

    clean_dir = tmp_path / "clean"
    clean_bundle, clean_learner = _bundle(clean_dir)
    label_path = Path(clean_bundle["label_split_sidecar_path"])
    label_path.chmod(0o644)
    label_path.write_text(label_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(exp.HashMismatchError, match="file_hash_mismatch"):
        exp.join_authority(
            clean_learner,
            label_path,
            clean_bundle["mutation_authority_sidecar_path"],
        )
    assert learner.matrix.shape == clean_learner.matrix.shape


def test_scenario_verify_6997_authority_join_keeps_features_separate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6997-JOIN returns routing without provenance in features."""
    bundle, learner = _bundle(tmp_path)
    joined = exp.join_authority(
        learner,
        bundle["label_split_sidecar_path"],
        bundle["mutation_authority_sidecar_path"],
    )

    assert np.array_equal(joined.feature_matrix, learner.matrix)
    assert joined.labels == ("equivalent", "non_equivalent")
    assert joined.splits == ("train", "held_out")
    assert all("mutation" in row for row in joined.mutation_authority_rows)
    assert "mutation" not in repr(joined.feature_matrix)

    label_rows, mutation_rows = _authority_rows(_source_rows())
    duplicate = [*label_rows, deepcopy(label_rows[0])]
    with pytest.raises(exp.AuthorityJoinError, match="duplicate_sidecar_key"):
        exp.join_authority_rows(learner, duplicate, mutation_rows)
    with pytest.raises(exp.AuthorityJoinError, match="sidecar_key_mismatch"):
        exp.join_authority_rows(learner, label_rows[:-1], mutation_rows)


def test_scenario_verify_6997_permutation_and_alpha_rename_are_invariant(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6997-INVARIANCE preserves tensors and fixed predictions."""
    source_rows = _source_rows()
    labels, mutations = _authority_rows(source_rows)
    base = exp.build_wide_learner_view(source_rows, expected_candidate_count=2)

    rng = random.Random(exp.RANDOM_SEED)
    rng.shuffle(source_rows)
    rng.shuffle(labels)
    rng.shuffle(mutations)
    renamed_labels, renamed_mutations = exp.alpha_rename_sidecars(labels, mutations)
    permuted = exp.build_wide_learner_view(source_rows, expected_candidate_count=2)

    base_bundle = exp.write_sidecar_bundle(
        tmp_path / "base",
        base["learner_rows"],
        labels,
        mutations,
        exp.FEATURE_ALLOWLIST,
    )
    changed_bundle = exp.write_sidecar_bundle(
        tmp_path / "changed",
        permuted["learner_rows"],
        renamed_labels,
        renamed_mutations,
        exp.FEATURE_ALLOWLIST,
    )
    base_batch = exp.load_learner_view(base_bundle["learner_view_path"], exp.FEATURE_ALLOWLIST)
    changed_batch = exp.load_learner_view(
        changed_bundle["learner_view_path"], exp.FEATURE_ALLOWLIST
    )

    assert base_batch.tensor_hash == changed_batch.tensor_hash
    assert base_batch.prediction_hash == changed_batch.prediction_hash
    assert np.array_equal(base_batch.matrix, changed_batch.matrix)
    assert renamed_mutations != mutations


def test_scenario_verify_6997_full_replay_is_ready_without_inference(tmp_path: Path) -> None:
    """REQ-VERIFY-6997 replays 138 candidates and 414 stored family rows."""
    artifact = exp.build_from_repo(REPO_ROOT, output_dir=tmp_path / "immutable")

    assert exp.validate_artifact(artifact) == []
    assert artifact["observed_candidate_count"] == 138
    assert artifact["observed_source_row_count"] == 414
    assert artifact["sidecar_rebuild_complete_score"] == 1
    assert artifact["blinded_learner_view_ready_score"] == 1
    assert artifact["gguf_inference_performed"] is False
    assert artifact["verifier_fit_performed"] is False
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("circular_positive:")


def test_scenario_verify_6997_missing_precondition_writes_blocked_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6997-PRECONDITIONS retains the exact blocked gate."""
    artifact = exp.build_from_repo(tmp_path, output_dir=tmp_path / "immutable")

    assert exp.validate_artifact(artifact) == []
    assert artifact["sidecar_rebuild_complete_score"] == 0
    assert artifact["blinded_learner_view_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_authority_sidecar_rebuild"
    assert artifact["gate_check_summary"]["failed_check"] == "source_artifact:exp6984"
    assert artifact["gate_check_summary"]["expected_value"] is True
    assert artifact["gate_check_summary"]["observed_value"] is False


def test_scenario_verify_6997_bare_scores_and_required_fields() -> None:
    """SCENARIO-VERIFY-6997-BARE rejects booleans and missing principles."""
    artifact = exp.empty_artifact(
        run_date="20260904",
        duration_s=0.0,
        checks=[exp.gate_check("blocked", True, False)],
    )
    assert exp.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["blinded_learner_view_ready_score"] = False
    bad["field_principles"].pop("rows")
    errors = exp.validate_artifact(bad)
    assert "score_not_bare_integer:blinded_learner_view_ready_score" in errors
    assert "field_principle_missing:rows" in errors

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert os.path.basename(exp.RESULT_PATH) == "experiment_6997_authority_sidecar_rebuild.json"


def test_scenario_verify_6997_defensive_row_guards_are_terminal() -> None:
    """SCENARIO-VERIFY-6997-ALLOWLIST rejects every malformed numeric row shape."""
    with pytest.raises(exp.RebuildError, match="semantic_hash_missing"):
        exp.candidate_key("prompt", "candidate")
    with pytest.raises(exp.RebuildError, match="candidate_count_mismatch"):
        exp.build_wide_learner_view([], expected_candidate_count=1)

    rows = _source_rows()
    nonnumeric = deepcopy(rows)
    nonnumeric[0][exp.SOURCE_FEATURES[0]] = "one"
    with pytest.raises(exp.RebuildError, match="nonnumeric_feature"):
        exp.build_wide_learner_view(nonnumeric, expected_candidate_count=2)

    same_semantics = deepcopy(rows[:3])
    for row in same_semantics:
        row["candidate_id"] = "candidate-copy"
    with pytest.raises(exp.RebuildError, match="duplicate_candidate_key"):
        exp.build_wide_learner_view([*rows[:3], *same_semantics], expected_candidate_count=2)

    clean = exp.build_wide_learner_view(rows, expected_candidate_count=2)["learner_rows"]
    changed_allowlist = exp.FEATURE_ALLOWLIST[:-1]
    assert exp.learner_schema_receipts(clean, changed_allowlist)["passed"] is False
    duplicate = [deepcopy(clean[0]), deepcopy(clean[0])]
    assert exp.learner_schema_receipts(duplicate, exp.FEATURE_ALLOWLIST)["prohibited_feature_rows"]
    deep_nested = deepcopy(clean)
    deep_nested[0]["metadata"] = {"nested": {"source": "secret"}}
    nested = exp.learner_schema_receipts(deep_nested, exp.FEATURE_ALLOWLIST)
    assert any(row["path"].endswith("metadata.nested") for row in nested["nested_metadata_rows"])
    short = deepcopy(clean)
    short[0]["features"].pop()
    assert exp.learner_schema_receipts(short, exp.FEATURE_ALLOWLIST)["prohibited_feature_rows"]
    categorical = deepcopy(clean)
    categorical[0]["features"][0] = "model-family"
    assert exp.learner_schema_receipts(categorical, exp.FEATURE_ALLOWLIST)[
        "categorical_identity_rows"
    ]
    with pytest.raises(exp.IsolationError, match="learner_schema_rejected"):
        exp._materialize_batch(categorical, exp.FEATURE_ALLOWLIST)


def test_scenario_verify_6997_immutable_conflicts_and_cleanup_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6997-HASH never overwrites changed immutable bytes."""
    path = tmp_path / "immutable.bin"
    exp._write_immutable(path, b"frozen")
    exp._write_immutable(path, b"frozen")
    with pytest.raises(exp.HashMismatchError, match="immutable_path_conflict"):
        exp._write_immutable(path, b"changed")

    target = tmp_path / "replace-fails.bin"

    def fail_replace(_self: Path, _target: Path) -> None:
        raise OSError("replace denied")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace denied"):
        exp._write_immutable(target, b"payload")
    assert not list(tmp_path.glob(".replace-fails.bin.*"))


def test_scenario_verify_6997_invalid_sidecar_schemas_fail_before_write(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6997-SIDECARS rejects bad schemas and key sets."""
    rebuilt = exp.build_wide_learner_view(_source_rows(), expected_candidate_count=2)
    labels, mutations = _authority_rows(_source_rows())
    invalid_labels = deepcopy(labels)
    invalid_labels[0]["extra"] = True
    with pytest.raises(exp.AuthorityJoinError, match="invalid_sidecar_schema"):
        exp.write_sidecar_bundle(
            tmp_path / "invalid",
            rebuilt["learner_rows"],
            invalid_labels,
            mutations,
            exp.FEATURE_ALLOWLIST,
        )

    with pytest.raises(exp.IsolationError, match="learner_schema_rejected"):
        exp.write_sidecar_bundle(
            tmp_path / "learner-invalid",
            [{"candidate_key": "key", "features": []}],
            [],
            [],
            exp.FEATURE_ALLOWLIST,
        )
    with pytest.raises(exp.AuthorityJoinError, match="sidecar_key_mismatch:write"):
        exp.write_sidecar_bundle(
            tmp_path / "key-mismatch",
            rebuilt["learner_rows"],
            labels[:-1],
            mutations[:-1],
            exp.FEATURE_ALLOWLIST,
        )


@pytest.mark.parametrize(
    ("manifest_change", "error"),
    [
        ({"kind": "label_split"}, "manifest_kind_mismatch"),
        ({"path": "/changed/learner_view.jsonl"}, "manifest_path_mismatch"),
        ({"row_count": 99}, "manifest_size_mismatch"),
        ({"ordered_key_hashes": []}, "manifest_key_mismatch"),
        ({"feature_allowlist": []}, "manifest_allowlist_mismatch"),
    ],
)
def test_scenario_verify_6997_manifest_mismatches_fail_closed(
    tmp_path: Path, manifest_change: dict[str, object], error: str
) -> None:
    """SCENARIO-VERIFY-6997-HASH validates every learner manifest binding."""
    bundle, _learner = _bundle(tmp_path)
    manifest_path = Path(str(bundle["learner_view_path"]) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_change)
    manifest_path.chmod(0o644)
    manifest_path.write_text(exp.canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(exp.HashMismatchError, match=error):
        exp.load_learner_view(bundle["learner_view_path"], exp.FEATURE_ALLOWLIST)


def test_scenario_verify_6997_loader_and_join_reject_all_extra_channels(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6997-LOADER rejects objects, allowlist drift, and wrong join paths."""
    bundle, learner = _bundle(tmp_path)
    with pytest.raises(exp.IsolationError, match="learner_path_required"):
        exp.load_learner_view(object(), exp.FEATURE_ALLOWLIST)  # type: ignore[arg-type]
    with pytest.raises(exp.IsolationError, match="frozen_allowlist_required"):
        exp.load_learner_view(bundle["learner_view_path"], exp.FEATURE_ALLOWLIST[:-1])
    with pytest.raises(exp.AuthorityJoinError, match="authority_sidecar_paths_required"):
        exp.join_authority(learner, bundle["learner_view_path"], bundle["learner_view_path"])


def test_scenario_verify_6997_selected_json_reader_handles_values_and_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6997-PRECONDITIONS reads selected escaped and scalar values."""
    path = tmp_path / "selected.json"
    payload = {
        "array": [{"text": 'quoted \\" value'}],
        "number": 7,
        "text": "slash \\\\ value",
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected = exp._selected_json_fields(path, ("array", "number", "text"))
    assert selected == payload
    with pytest.raises(exp.RebuildError, match="source_field_missing"):
        exp._selected_json_fields(path, ("absent",))


def test_scenario_verify_6997_source_disagreements_are_never_repaired() -> None:
    """SCENARIO-VERIFY-6997-PRECONDITIONS rejects stored source disagreements."""
    bank = {
        "per_candidate_model_rows": [
            {"candidate_id": "candidate", "model_id": "model", exp.SEQUENCE_FEATURES[0]: 1}
        ],
        "sequence_feature_rows": [],
        "parser_feature_rows": [],
    }
    with pytest.raises(exp.RebuildError, match="stored_feature_component_missing"):
        exp._source_model_rows(bank)

    sequence = {field: 1 for field in exp.SEQUENCE_FEATURES}
    feature = {"candidate_id": "candidate", "model_id": "model", **sequence}
    sequence[exp.SEQUENCE_FEATURES[0]] = 2
    bank = {
        "per_candidate_model_rows": [feature],
        "sequence_feature_rows": [{"candidate_id": "candidate", "model_id": "model", **sequence}],
        "parser_feature_rows": [{"candidate_id": "candidate", "model_id": "model"}],
    }
    with pytest.raises(exp.RebuildError, match="stored_sequence_disagreement"):
        exp._source_model_rows(bank)

    with pytest.raises(exp.RebuildError, match="conflicting_joined_label"):
        exp._collapse_labels(
            {
                "joined_label_rows": [
                    {"candidate_id": "candidate", "exact_label": "yes", "source_block": "a"},
                    {"candidate_id": "candidate", "exact_label": "no", "source_block": "a"},
                ]
            }
        )


def _minimal_authority_sources() -> dict[str, object]:
    return {
        "exp6986": {"joined_label_rows": []},
        "exp6984": {
            "per_candidate_rows": [],
            "mutation_attempt_rows": [],
            "z3_authority_rows": [],
            "enumeration_authority_rows": [],
            "authority_agreement_rows": [],
        },
        "exp6985": {
            "per_candidate_rows": [],
            "per_event_results": [],
            "authority_agreement_rows": [],
        },
        "exp6976": {
            "per_candidate_rows": [],
            "exact_witness_rows": [],
            "solver_agreement_rows": [],
        },
        "exp6987": {"per_candidate_model_rows": []},
    }


def test_scenario_verify_6997_authority_metadata_disagreements_fail() -> None:
    """SCENARIO-VERIFY-6997-JOIN rejects metadata conflicts, gaps, and unknown sources."""
    sources = _minimal_authority_sources()
    sources["exp6987"]["per_candidate_model_rows"] = [  # type: ignore[index]
        {"candidate_id": "candidate", "source_block": "one"},
        {"candidate_id": "candidate", "source_block": "two"},
    ]
    with pytest.raises(exp.RebuildError, match="source_metadata_conflict"):
        exp._authority_sidecar_rows(sources, [])

    sources = _minimal_authority_sources()
    with pytest.raises(exp.RebuildError, match="authority_metadata_missing"):
        exp._authority_sidecar_rows(
            sources, [{"candidate_id": "candidate", "candidate_key": "key"}]
        )

    sources = _minimal_authority_sources()
    sources["exp6986"]["joined_label_rows"] = [  # type: ignore[index]
        {"candidate_id": "candidate", "exact_label": "yes", "source_block": "unknown"}
    ]
    sources["exp6987"]["per_candidate_model_rows"] = [  # type: ignore[index]
        {"candidate_id": "candidate", "source_block": "unknown"}
    ]
    with pytest.raises(exp.RebuildError, match="unknown_source_block"):
        exp._authority_sidecar_rows(
            sources, [{"candidate_id": "candidate", "candidate_key": "key"}]
        )


def test_scenario_verify_6997_path_and_terminal_precondition_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6997-PRECONDITIONS preserves path and terminal failures."""
    monkeypatch.setattr(exp.tempfile, "mkstemp", lambda **_kwargs: (_ for _ in ()).throw(OSError()))
    assert exp._path_writable(tmp_path / "denied") is False

    passed = {
        "checks": [exp.gate_check("hashes", True, True)],
        "source_artifact_hashes": {},
        "passed": True,
    }
    monkeypatch.setattr(exp, "collect_preconditions", lambda *_args, **_kwargs: passed)
    monkeypatch.setattr(exp, "_read_jsonl", lambda _path: [])
    monkeypatch.setattr(
        exp,
        "_selected_json_fields",
        lambda _path, fields: {field: ([] if field.endswith("rows") else 0) for field in fields},
    )
    artifact = exp.build_from_repo(tmp_path, output_dir=tmp_path / "unused")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "exp6984_terminal"


def test_scenario_verify_6997_artifact_validator_and_run_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-6997-BARE validates every terminal field and atomic write."""
    valid = exp.empty_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.0,
        checks=[exp.gate_check("blocked", True, False)],
    )
    bad = deepcopy(valid)
    bad.pop("rows")
    bad["field_principles"] = []
    bad["inference_substrate"] = "wrong"
    bad["gguf_inference_performed"] = True
    bad["verifier_fit_performed"] = True
    bad["verifier_is_oracle"] = False
    bad["honest_verdict"] = "wrong"
    errors = exp.validate_artifact(bad)
    assert "required_field_missing:rows" in errors
    assert "field_principles_not_mapping" in errors
    assert "inference_substrate_mismatch" in errors
    assert "gguf_inference_must_be_false" in errors
    assert "verifier_fit_must_be_false" in errors
    assert "verifier_is_oracle_must_be_true" in errors
    assert "verdict_prefix_mismatch" in errors

    output = tmp_path / "result.json"
    monkeypatch.setattr(exp, "build_from_repo", lambda *_args, **_kwargs: valid)
    returned = exp.run(repo_root=tmp_path, output_path=output, output_dir=tmp_path / "data")
    assert returned == valid
    assert json.loads(output.read_text(encoding="utf-8")) == valid

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["bad"])
    with pytest.raises(exp.RebuildError, match="artifact_validation_failed:bad"):
        exp.run(repo_root=tmp_path, output_path=output, output_dir=tmp_path / "data")

    def fail_replace(_self: Path, _target: Path) -> None:
        raise OSError("result replace denied")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="result replace denied"):
        exp._write_result(tmp_path / "replace-fails.json", valid)
    assert not list(tmp_path.glob(".replace-fails.json.*"))
