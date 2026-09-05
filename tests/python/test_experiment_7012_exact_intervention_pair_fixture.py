"""Tests for the exact minimal intervention-pair fixture.

Spec refs: REQ-VERIFY-7012 and SCENARIO-VERIFY-7012-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7012_exact_intervention_pair_fixture as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def complete_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build one real fresh-process fixture for all read-only assertions."""

    root = tmp_path_factory.mktemp("exp7012-complete")
    return exp.run_controller(
        repo_root=REPO_ROOT,
        result_path=root / "experiment_7012.json",
        data_root=root / "raw",
        run_date=exp.RUN_DATE,
    )


def _clean_pair() -> dict[str, object]:
    """Return one frozen clean pair without sharing mutable test state."""

    manifest = exp.build_source_manifest()
    return deepcopy(exp.frozen_clean_pairs()[manifest[0]["source_pair_id"]])


def _valid_block_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create a small balanced block for negative audit tests."""

    learner = [
        {
            "semantic_key": f"sha256:{index:064x}",
            "neutral_block_position": index % 2,
            "prompt": "candidate" + (" " * 4),
        }
        for index in range(4)
    ]
    sidecar = [
        {
            "semantic_key": row["semantic_key"],
            "exact_label": "equivalent" if index in {0, 3} else "non_equivalent",
            "serialization_template": "compact" if index < 2 else "expanded",
            "mutation_kind": "bound_change",
            "source_group_id": "source-0",
            "source_family": "family-0",
            "split": "train",
            "neutral_block_position": row["neutral_block_position"],
        }
        for index, row in enumerate(learner)
    ]
    return learner, sidecar


def _fake_certificate(_pair: object, candidate_id: str) -> dict[str, object]:
    """Return a terminal dual-authority receipt for fast controller-logic tests."""

    changed = candidate_id.endswith("changed")
    counterexamples = {"assignment": {"x": 0}} if changed else {}
    relation = "non_equivalent" if changed else "equivalent"
    return {
        "agreement": {
            "all_required_agreement": True,
            "certified_relation": relation,
        },
        "enumeration": {"counterexamples": counterexamples, "witnesses": {}},
        "z3": {"counterexamples": counterexamples, "status": "sat"},
    }


def _writable_source_root(root: Path) -> Path:
    """Create harmless stand-in source files for the sandbox write probe."""

    for relative in exp.SOURCE_PATHS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    return root


def test_required_schema_principles_and_fixed_counts() -> None:
    """REQ-VERIFY-7012: Each artifact field has one scientific reason."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)
    assert exp.EXPECTED_PAIR_COUNT == 48
    assert exp.EXPECTED_FAMILY_COUNT == 4
    assert exp.INFERENCE_SUBSTRATE == "deterministic_exact_pair_fixture_no_llm"
    assert len(exp.SOURCE_FAMILIES) == len(exp.MUTATION_KINDS) == 4


def test_preconditions_require_sources_four_families_exact_engines_and_writes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7012-PRECONDITIONS: Every exact dependency is checked."""

    checks = exp.collect_preconditions(REPO_ROOT, tmp_path / "rows")
    assert all(row["passed"] for row in checks)
    missing = exp.collect_preconditions(tmp_path / "missing", tmp_path / "other")
    assert exp.gate_summary(missing)["failed_check"] == "source_hash:exp6984"
    assert exp.gate_summary(missing)["expected_value"].startswith("sha256:")
    assert exp.gate_summary(missing)["observed_value"] is None


def test_unwritable_fixture_path_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-7012-PRECONDITIONS: A write failure blocks the fixture."""

    monkeypatch.setattr(
        exp.tempfile, "mkstemp", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    assert exp.path_is_writable(tmp_path / "rows") is False


@pytest.mark.parametrize("mutation_kind", exp.MUTATION_KINDS)
def test_each_mutation_is_one_exact_structural_edit(mutation_kind: str) -> None:
    """SCENARIO-VERIFY-7012-MINIMALITY: Each declared intervention is minimal."""

    clean = _clean_pair()
    changed, mutation = exp.apply_intervention(clean, mutation_kind)
    receipt = exp.minimality_receipt(clean, changed, mutation, direction="violation")
    assert mutation["changed_paths"]
    assert receipt["edit_distance"] == 1
    assert receipt["minimal"] is True


def test_noop_multi_edit_and_nonminimal_repair_are_rejected() -> None:
    """SCENARIO-VERIFY-7012-MINIMALITY: Zero or multiple edits cannot enter."""

    clean = _clean_pair()
    no_op = exp.minimality_receipt(
        clean,
        clean,
        {"mutation_kind": "bound_change", "changed_paths": []},
        direction="violation",
    )
    assert no_op["minimal"] is False
    changed, mutation = exp.apply_intervention(clean, "bound_change")
    changed["target"]["objective"]["direction"] = "max"
    multi = exp.minimality_receipt(clean, changed, mutation, direction="violation")
    repair = exp.minimality_receipt(changed, clean, mutation, direction="repair")
    assert multi["edit_distance"] > 1 and multi["minimal"] is False
    assert repair["edit_distance"] > 1 and repair["minimal"] is False


def test_piecewise_coefficients_and_type_changes_have_exact_edit_distance() -> None:
    """REQ-VERIFY-7012: Piecewise terms and JSON type changes use the same metric."""

    manifest = exp.build_source_manifest()
    row = next(row for row in manifest if "piecewise" in row["source_family"])
    clean = deepcopy(exp.frozen_clean_pairs()[row["source_pair_id"]])
    changed, mutation = exp.apply_intervention(clean, "objective_coefficient_change")
    assert exp.minimality_receipt(clean, changed, mutation, direction="violation")["minimal"]
    assert exp.structural_edit_distance(1, "1") == 1
    with pytest.raises(exp.FixtureError, match="unknown_serialization_template"):
        exp._serialize_prompt(clean, "unknown")


def test_unknown_mutation_kind_is_rejected() -> None:
    """REQ-VERIFY-7012: Mutation support uses a closed executable family set."""

    with pytest.raises(exp.FixtureError, match="unknown_mutation_kind"):
        exp.apply_intervention(_clean_pair(), "unknown")


def test_ambiguous_or_mismatched_authority_outcomes_reject_the_pair() -> None:
    """SCENARIO-VERIFY-7012-MINIMALITY: Exact authorities must decide together."""

    clean = _clean_pair()
    changed, _ = exp.apply_intervention(clean, "objective_direction_reversal")
    clean_certificate = exp.certify_candidate(clean, "clean")
    changed_certificate = exp.certify_candidate(changed, "changed")
    assert exp.authority_pair_receipt(clean_certificate, changed_certificate)["passed"] is True
    ambiguous = deepcopy(changed_certificate)
    ambiguous["agreement"]["all_required_agreement"] = False
    ambiguous["z3"]["status"] = "unknown"
    assert exp.authority_pair_receipt(clean_certificate, ambiguous)["passed"] is False
    wrong = deepcopy(changed_certificate)
    wrong["agreement"]["certified_relation"] = "equivalent"
    assert exp.authority_pair_receipt(clean_certificate, wrong)["passed"] is False


def test_isomorphism_requires_normalized_structure_and_exact_label_invariance() -> None:
    """SCENARIO-VERIFY-7012-ISOMORPHISM: Renaming cannot change semantics."""

    clean = _clean_pair()
    renamed, rename = exp.alpha_rename_pair(clean, "block-0")
    valid = exp.isomorphism_receipt(clean, renamed, "equivalent", "equivalent", rename)
    assert valid["passed"] is True
    invalid = deepcopy(renamed)
    invalid["target"]["constraints"][0]["rhs"] = "314159"
    assert (
        exp.isomorphism_receipt(clean, invalid, "equivalent", "equivalent", rename)["passed"]
        is False
    )
    assert (
        exp.isomorphism_receipt(clean, renamed, "equivalent", "non_equivalent", rename)["passed"]
        is False
    )


def test_semantic_keys_are_unique_and_row_reordering_is_byte_stable() -> None:
    """SCENARIO-VERIFY-7012-KEYS: Canonical learner bytes ignore input order."""

    learner, _ = _valid_block_rows()
    assert exp.canonical_learner_bytes(learner) == exp.canonical_learner_bytes(
        list(reversed(learner))
    )
    duplicate = [*learner, deepcopy(learner[0])]
    with pytest.raises(exp.FixtureError, match="duplicate_semantic_key"):
        exp.canonical_learner_bytes(duplicate)


def test_block_audit_rejects_label_length_serialization_and_source_imbalance() -> None:
    """SCENARIO-VERIFY-7012-BALANCE: Every nuisance check is independently binding."""

    learner, sidecar = _valid_block_rows()
    assert exp.audit_matched_block("block-0", learner, sidecar)["passed"] is True

    imbalanced = deepcopy(sidecar)
    imbalanced[1]["exact_label"] = "equivalent"
    assert exp.audit_matched_block("block-0", learner, imbalanced)["label_balanced"] is False

    length_drift = deepcopy(learner)
    length_drift[0]["prompt"] += "x"
    assert exp.audit_matched_block("block-0", length_drift, sidecar)["length_balanced"] is False

    template_drift = deepcopy(sidecar)
    template_drift[0]["serialization_template"] = "expanded"
    assert (
        exp.audit_matched_block("block-0", learner, template_drift)["serialization_balanced"]
        is False
    )

    source_drift = deepcopy(sidecar)
    source_drift[0]["source_group_id"] = "other"
    assert exp.audit_matched_block("block-0", learner, source_drift)["source_fixed"] is False


def test_direct_nested_and_normalized_metadata_aliases_are_denied() -> None:
    """SCENARIO-VERIFY-7012-LEAKAGE: Metadata aliases cannot enter learner rows."""

    learner, _ = _valid_block_rows()
    assert exp.validate_learner_rows(learner) == []
    direct = deepcopy(learner)
    direct[0]["oracle-label"] = "equivalent"
    nested = deepcopy(learner)
    nested[0]["payload"] = {"splitAlias": "train"}
    assert any("oracle-label" in row["path"] for row in exp.validate_learner_rows(direct))
    assert any("splitAlias" in row["path"] for row in exp.validate_learner_rows(nested))


def test_invalid_learner_values_lists_and_sidecar_duplicates_are_denied() -> None:
    """SCENARIO-VERIFY-7012-LEAKAGE: Invalid primitives and nested lists fail closed."""

    learner, sidecar = _valid_block_rows()
    invalid = deepcopy(learner)
    invalid[0] = {
        "semantic_key": 1,
        "neutral_block_position": True,
        "prompt": "",
        "payload": [{"authorityAlias": "x"}],
    }
    reasons = {row["reason"] for row in exp.validate_learner_rows(invalid)}
    assert {"invalid_key", "invalid_position", "invalid_prompt"} <= reasons
    with pytest.raises(exp.FixtureError, match="prohibited_learner_field"):
        exp.canonical_learner_bytes(invalid)
    with pytest.raises(exp.FixtureError, match="duplicate_sidecar_key"):
        exp._canonical_sidecar_bytes([*sidecar, deepcopy(sidecar[0])])


def test_learner_loader_rejects_sidecar_access_and_authority_bearing_bytes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7012-LEAKAGE: Loader access stops at the learner file."""

    learner, _ = _valid_block_rows()
    path = tmp_path / "learner.jsonl"
    path.write_bytes(exp.canonical_learner_bytes(learner))
    loaded = exp.load_learner_prompts(path)
    assert [row["semantic_key"] for row in loaded] == sorted(row["semantic_key"] for row in learner)
    with pytest.raises(exp.IsolationError, match="sidecar_access_denied"):
        exp.load_learner_prompts(path, sidecar_path=tmp_path / "authority.jsonl")
    path.write_text(json.dumps({**learner[0], "authorityWitness": {}}) + "\n")
    with pytest.raises(exp.IsolationError, match="prohibited_learner_field"):
        exp.load_learner_prompts(path)


def test_learner_loader_rejects_nonobjects_invalid_json_and_missing_files(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7012-LEAKAGE: Malformed learner storage fails closed."""

    path = tmp_path / "learner.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(exp.IsolationError, match="learner_row_not_object"):
        exp.load_learner_prompts(path)
    path.write_text("{\n", encoding="utf-8")
    with pytest.raises(exp.IsolationError, match="learner_file_invalid:JSONDecodeError"):
        exp.load_learner_prompts(path)
    with pytest.raises(exp.IsolationError, match="learner_file_invalid:FileNotFoundError"):
        exp.load_learner_prompts(tmp_path / "missing.jsonl")


def test_sidecar_permutation_replacement_deletion_and_alpha_rename_are_invariant() -> None:
    """SCENARIO-VERIFY-7012-SIDECARS: Sidecar state has no learner effect."""

    learner, sidecar = _valid_block_rows()
    rows = exp.sidecar_invariance_receipts(learner, sidecar)
    assert {row["condition"] for row in rows} == {
        "correct",
        "permuted",
        "replaced",
        "deleted",
        "alpha_renamed",
    }
    assert all(row["passed"] for row in rows)
    assert len({row["learner_prompt_hash"] for row in rows}) == 1


def test_source_manifest_freezes_counts_splits_and_groups_before_scoring() -> None:
    """SCENARIO-VERIFY-7012-SPLITS: Source groups define whole-block partitions."""

    rows = exp.build_source_manifest()
    assert len(rows) == 48
    assert len({row["source_group_id"] for row in rows}) == 48
    assert {row["source_family"] for row in rows} == set(exp.SOURCE_FAMILIES)
    assert {split: sum(row["split"] == split for row in rows) for split in exp.SPLITS} == {
        "train": 24,
        "calibration": 8,
        "held_source": 8,
        "sealed_headroom": 8,
    }
    assert all(row["frozen_before_authority_scoring"] for row in rows)


def test_complete_artifact_replays_all_counts_balances_files_and_verdict(
    complete_artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-7012: The fresh-process fixture meets every release gate."""

    artifact = complete_artifact
    assert exp.validate_artifact(artifact) == []
    assert artifact["expected_pair_count"] == artifact["observed_pair_count"] == 48
    assert artifact["expected_family_count"] == artifact["observed_family_count"] == 4
    assert len(artifact["block_rows"]) == len(artifact["pair_rows"]) == 48
    assert len(artifact["sidecar_intervention_rows"]) == 192
    assert artifact["rejected_block_rows"] == []
    assert artifact["intervention_pair_fixture_ready_score"] == 1
    assert type(artifact["intervention_pair_fixture_ready_score"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("circular_positive:")
    assert Path(artifact["learner_prompt_path"]).is_file()
    assert Path(artifact["authority_sidecar_path"]).is_file()
    assert exp.sha256_path(Path(artifact["learner_prompt_path"])) == artifact["learner_prompt_hash"]
    assert (
        exp.sha256_path(Path(artifact["authority_sidecar_path"]))
        == artifact["authority_sidecar_hash"]
    )
    assert all(row["passed"] for row in artifact["nuisance_balance_rows"])
    assert all(row["passed"] for row in artifact["isomorphism_rows"])
    assert all(row["minimal"] for row in artifact["minimality_rows"])
    assert all(row["passed"] for row in artifact["authority_witness_rows"])


def test_direct_builder_covers_success_and_rejected_denominators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-7012: The in-process reducer preserves accepted and rejected rows."""

    monkeypatch.setattr(exp, "certify_candidate", _fake_certificate)
    artifact = exp.build_artifact(
        REPO_ROOT,
        tmp_path / "accepted",
        duration_s=0.0,
        fresh_process_receipt={"passed": True},
    )
    assert artifact["intervention_pair_fixture_ready_score"] == 1
    assert exp.validate_artifact(artifact) == []

    real_audit = exp.audit_matched_block

    def reject_audit(*args: object, **kwargs: object) -> dict[str, object]:
        receipt = real_audit(*args, **kwargs)  # type: ignore[arg-type]
        receipt["passed"] = False
        return receipt

    monkeypatch.setattr(exp, "audit_matched_block", reject_audit)
    rejected = exp.build_artifact(
        REPO_ROOT,
        tmp_path / "rejected",
        duration_s=0.0,
        fresh_process_receipt={"passed": True},
    )
    assert len(rejected["rejected_block_rows"]) == 48
    assert rejected["intervention_pair_fixture_ready_score"] == 0

    blocked = exp.build_artifact(
        tmp_path / "missing",
        tmp_path / "blocked",
        duration_s=0.0,
        fresh_process_receipt={"passed": False},
    )
    assert blocked["verdict_class"] == "blocked"


def test_blocked_artifact_is_schema_complete_and_names_first_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7012-PRECONDITIONS: Missing sources produce one blocked result."""

    checks = exp.collect_preconditions(tmp_path / "missing", tmp_path / "rows")
    artifact = exp.build_blocked_artifact(
        repo_root=tmp_path / "missing",
        data_root=tmp_path / "rows",
        preconditions=checks,
        duration_s=0.1,
    )
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["intervention_pair_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_intervention_pair_fixture")
    assert artifact["gate_check_summary"]["failed_check"]
    assert exp.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value.pop("rows"), "missing_required_field"),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles"),
        (
            lambda value: value.__setitem__("intervention_pair_fixture_ready_score", True),
            "bare_ready_score",
        ),
        (lambda value: value["pair_rows"].pop(), "pair_count"),
        (lambda value: value.__setitem__("observed_family_count", 3), "family_count"),
        (lambda value: value["nuisance_balance_rows"][0].__setitem__("passed", False), "nuisance"),
        (lambda value: value.__setitem__("learner_prompt_hash", "sha256:forged"), "learner_hash"),
        (
            lambda value: value.__setitem__("authority_sidecar_hash", "sha256:forged"),
            "sidecar_hash",
        ),
        (lambda value: value.__setitem__("verifier_is_oracle", False), "verifier_is_oracle"),
        (lambda value: value.__setitem__("verdict_class", "positive"), "verdict_consistency"),
        (lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"), "checksum"),
        (lambda value: value.__setitem__("inference_substrate", "llm"), "inference_substrate"),
        (lambda value: value.__setitem__("verdict_class", "mystery"), "verdict_class"),
        (lambda value: value.__setitem__("honest_verdict", "wrong"), "verdict_prefix"),
        (lambda value: value["minimality_rows"].pop(), "exact_minimality"),
        (lambda value: value["group_split_rows"].pop(), "group_splits"),
        (lambda value: value["sidecar_intervention_rows"].pop(), "sidecar_rows"),
        (
            lambda value: value["prohibited_feature_rows"][0].__setitem__("passed", False),
            "prohibited_features",
        ),
        (
            lambda value: value["gate_check_summary"].__setitem__("passed", False),
            "ready_score_consistency",
        ),
    ],
)
def test_validator_rejects_forged_rows_hashes_and_readiness(
    complete_artifact: dict[str, object], mutate: object, expected: str
) -> None:
    """SCENARIO-VERIFY-7012-ARTIFACT: Readiness is recomputed from evidence."""

    artifact = deepcopy(complete_artifact)
    mutate(artifact)  # type: ignore[operator]
    assert any(expected in error for error in exp.validate_artifact(artifact))


def test_immutable_writer_and_fixed_date_command_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7012-ARTIFACT: Frozen bytes and execution date cannot drift."""

    path = tmp_path / "frozen.jsonl"
    first = exp.write_immutable(path, b"one\n")
    assert exp.write_immutable(path, b"one\n") == first
    with pytest.raises(exp.ImmutableFixtureError, match="immutable_fixture_mismatch"):
        exp.write_immutable(path, b"two\n")
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.run_controller(
            repo_root=REPO_ROOT,
            result_path=tmp_path / "wrong.json",
            data_root=tmp_path / "wrong-raw",
            run_date="20260904",
        )


def test_blocked_validator_recomputes_score_gate_and_prefix(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7012-ARTIFACT: Blocked verdict fields remain binding."""

    blocked = exp.build_blocked_artifact(
        repo_root=tmp_path,
        data_root=tmp_path / "raw",
        preconditions=[exp.gate_check("x", True, False)],
        duration_s=0.0,
    )
    for field, value, expected in (
        ("intervention_pair_fixture_ready_score", 1, "blocked_ready_score"),
        ("gate_check_summary", {"passed": True, "failed_check": None}, "blocked_gate_summary"),
        ("honest_verdict", "blocked: other", "blocked_verdict"),
    ):
        forged = deepcopy(blocked)
        forged[field] = value
        assert expected in exp.validate_artifact(forged)


def test_sandbox_command_atomic_cleanup_and_controller_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    complete_artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7012-PRECONDITIONS: Isolation and controller failures block."""

    assert (
        exp.sandbox_receipt(_writable_source_root(tmp_path / "writable"))["source_tree_read_only"]
        is False
    )
    monkeypatch.setenv("CARNOT_EXP7012_PARENT_NETNS", "net:[1]")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(exp.os, "readlink", lambda _path: "net:[2]")
    monkeypatch.setattr(exp.Path, "glob", lambda _self, _pattern: [])
    monkeypatch.setattr(
        exp.Path,
        "open",
        lambda _self, *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()),
    )
    assert exp.sandbox_receipt(REPO_ROOT)["passed"] is True
    command = exp.fresh_process_command(
        executable=Path("python"),
        wrapper=Path("wrapper.py"),
        repo_root=REPO_ROOT,
        data_root=tmp_path,
        writable_root=tmp_path,
        output_path=tmp_path / "out.json",
        run_date=exp.RUN_DATE,
    )
    assert "--unshare-net" in command and command[-1].endswith("out.json")
    monkeypatch.undo()

    monkeypatch.setattr(exp.shutil, "which", lambda _name: None)
    blocked = exp.run_controller(
        repo_root=REPO_ROOT,
        result_path=tmp_path / "no-bwrap.json",
        data_root=tmp_path / "no-bwrap",
    )
    assert blocked["verdict_class"] == "blocked"
    monkeypatch.undo()

    monkeypatch.setattr(exp.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(
        exp.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=7)
    )
    blocked = exp.run_controller(
        repo_root=REPO_ROOT,
        result_path=tmp_path / "child-failed.json",
        data_root=tmp_path / "child-failed",
    )
    assert blocked["gate_check_summary"]["failed_check"] == "fresh_process_exit_code"
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(exp.FixtureError, match="artifact_validation_failed:forced"):
        exp.run_controller(
            repo_root=REPO_ROOT,
            result_path=tmp_path / "child-invalid.json",
            data_root=tmp_path / "child-invalid",
        )
    monkeypatch.undo()

    def successful_child(command: list[str], **_kwargs: object) -> SimpleNamespace:
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(complete_artifact), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    source_calls = 0

    def drifting_hashes(_root: Path) -> dict[str, str]:
        nonlocal source_calls
        source_calls += 1
        return {"source": "before" if source_calls == 1 else "after"}

    monkeypatch.setattr(exp.subprocess, "run", successful_child)
    monkeypatch.setattr(exp, "source_artifact_hashes", drifting_hashes)
    drifted = exp.run_controller(
        repo_root=REPO_ROOT,
        result_path=tmp_path / "drifted.json",
        data_root=tmp_path / "drifted",
    )
    assert drifted["gate_check_summary"]["failed_check"] == "source_hashes_unchanged"


def test_atomic_writer_cleans_temporary_file_on_publish_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-7012: Failed atomic publication does not leave a partial result."""

    original_unlink = exp.os.unlink
    monkeypatch.setattr(
        exp.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("publish failed")),
    )
    with pytest.raises(OSError, match="publish failed"):
        exp.write_json_atomic(tmp_path / "result.json", {"value": 1})
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(
        exp.os,
        "unlink",
        lambda *_args: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    with pytest.raises(OSError, match="publish failed"):
        exp.write_json_atomic(tmp_path / "result.json", {"value": 2})
    monkeypatch.setattr(exp.os, "unlink", original_unlink)
    for path in tmp_path.iterdir():
        path.unlink()


def test_remaining_fail_closed_branches_are_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-7012: Serialization, preflight, sandbox, and publish guards execute."""

    clean = _clean_pair()
    with pytest.raises(exp.FixtureError, match="unknown_serialization_template"):
        exp._serialize_prompt(clean, "unknown")
    blocked = exp.build_artifact(
        tmp_path / "missing",
        tmp_path / "blocked",
        duration_s=0.0,
        fresh_process_receipt={"passed": False},
    )
    assert blocked["verdict_class"] == "blocked"
    assert (
        exp.sandbox_receipt(_writable_source_root(tmp_path / "writable"))["source_tree_read_only"]
        is False
    )

    monkeypatch.setattr(exp.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced"])
    with pytest.raises(exp.FixtureError, match="artifact_validation_failed:forced"):
        exp.run_controller(
            repo_root=REPO_ROOT,
            result_path=tmp_path / "invalid.json",
            data_root=tmp_path / "invalid",
        )
