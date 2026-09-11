"""Tests for the V635 query-refinement fixture.

Spec refs: REQ-CL-7212 and SCENARIO-CL-7212-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7212_v635_refinement_fixture as exp
from carnot.memory.transactional_constraint_memory import TransactionalConstraintMemory


def _public(event_id: str, family: str, value: int) -> dict[str, object]:
    """Build one learner-visible event without authority fields."""

    return {
        "event_id": event_id,
        "seed": exp.STREAM_SEEDS[0],
        "chronology_index": 40,
        "family_id": family,
        "numeric_value": value,
        "public_input": f"family={family};value={value}",
    }


def test_contract_constants_match_the_frozen_task() -> None:
    """REQ-CL-7212: Freeze exact families, seeds, phases, and budgets."""

    assert exp.STREAM_SEEDS == tuple(range(7_212_001, 7_212_021))
    assert exp.FAMILIES == (
        "lower_bound",
        "upper_bound",
        "modular_equals",
        "cyclic_window",
    )
    assert exp.PARAMETER_DOMAIN == tuple(range(33))
    assert [phase["name"] for phase in exp.PHASES] == [
        "stable",
        "drift",
        "recurrence",
        "poison",
    ]
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.QUERY_BUDGET == 64
    assert exp.FITTING_BUDGET == 48
    assert exp.VALIDATION_BUDGET == 16
    assert exp.PENDING_CAPACITY == 4


def test_actual_principle_wrapper_is_the_only_unwrapped_mapping() -> None:
    """SCENARIO-CL-7212-PRECONDITIONS: Do not unwrap arbitrary dictionaries."""

    wrapped = {"principle": "explain", "value": {"passed": True}}
    arbitrary = {"value": 1, "other": 2}
    extra = {"principle": "explain", "value": 1, "evidence": "x"}
    assert exp.unwrap_principled(wrapped) == {"passed": True}
    assert exp.unwrap_principled(arbitrary) is arbitrary
    assert exp.unwrap_principled(extra) is extra


def test_evaluator_worker_seals_separate_conformant_views(tmp_path: Path) -> None:
    """SCENARIO-CL-7212-STREAM/AUDIT: Keep all authority bytes separate."""

    paths = exp.ExperimentPaths.under(tmp_path)
    assert exp.evaluator_worker(paths, seeds=(exp.STREAM_SEEDS[0],)) == 0
    public = exp.read_jsonl(paths.public_stream)
    authority = exp.read_jsonl(paths.authority_sidecar)
    warmup = exp.read_jsonl(paths.released_warmup)
    manifest = json.loads(paths.split_feedback_manifest.read_text(encoding="utf-8"))

    assert len(public) == exp.EVENTS_PER_SEED
    assert len([row for row in authority if row["row_type"] == "event_authority"]) == 1024
    assert len([row for row in authority if row["row_type"] == "audit_label"]) == 528
    assert len(warmup) == exp.WARMUP_COUNT
    assert exp.public_leakage_errors(public) == []
    assert (
        exp.stream_conformance_errors(
            public,
            authority,
            warmup,
            manifest,
            expected_seeds=(exp.STREAM_SEEDS[0],),
        )
        == []
    )
    assert all("exact_label" not in row for row in public)
    assert all(row["source"] == "released_queried_feedback" for row in warmup)
    for partition in manifest["validation_partitions"]:
        assert set(partition["reserved_validation_x"]).isdisjoint(partition["fitting_x_domain"])

    with pytest.raises(exp.ImmutableSealError, match="immutable_path_conflict"):
        exp.write_immutable(paths.public_stream, b"changed\n")
    exp.write_immutable(paths.public_stream, paths.public_stream.read_bytes())
    assert exp._phase_for(300)["name"] == "drift"

    assert "event_count" in exp.stream_conformance_errors(
        public[:-1], authority, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    leaked = deepcopy(public)
    leaked[0]["nested"] = {"hidden_parameter": 1}
    assert "public_authority_leak" in exp.stream_conformance_errors(
        leaked, authority, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    duplicate = deepcopy(public)
    duplicate[-1]["event_id"] = duplicate[0]["event_id"]
    assert "public_event_identity" in exp.stream_conformance_errors(
        duplicate, authority, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    unmatched = deepcopy(public)
    unmatched[-1]["event_id"] = "unmatched-public-id"
    assert "public_authority_identity" in exp.stream_conformance_errors(
        unmatched, authority, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    bad_event = deepcopy(authority)
    next(row for row in bad_event if row["row_type"] == "event_authority")["exact_label"] = "bad"
    assert "event_exact_grounding" in exp.stream_conformance_errors(
        public, bad_event, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    bad_audit = deepcopy(authority)
    next(row for row in bad_audit if row["row_type"] == "audit_label")["scoring_only"] = False
    assert "audit_exact_grounding" in exp.stream_conformance_errors(
        public, bad_audit, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    assert "audit_panel_count" in exp.stream_conformance_errors(
        public, authority[:-1], warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    assert "warmup_count" in exp.stream_conformance_errors(
        public, authority, warmup[:-1], manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    bad_warmup = deepcopy(warmup)
    bad_warmup[0]["source"] = "future"
    assert "warmup_release" in exp.stream_conformance_errors(
        public, authority, bad_warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    short_manifest = deepcopy(manifest)
    short_manifest["validation_partitions"] = short_manifest["validation_partitions"][:-1]
    assert "validation_partition_count" in exp.stream_conformance_errors(
        public, authority, warmup, short_manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    bad_manifest = deepcopy(manifest)
    bad_manifest["validation_partitions"][0]["fitting_x_domain"].append(
        bad_manifest["validation_partitions"][0]["reserved_validation_x"][0]
    )
    assert "validation_partition" in exp.stream_conformance_errors(
        public, authority, warmup, bad_manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    assert "seed_event_count" in exp.stream_conformance_errors(
        public,
        authority,
        warmup,
        manifest,
        expected_seeds=(exp.STREAM_SEEDS[0], exp.STREAM_SEEDS[1]),
    )
    unbalanced = deepcopy(public)
    unbalanced[0]["family_id"] = "upper_bound"
    assert "phase_family_balance" in exp.stream_conformance_errors(
        unbalanced, authority, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )
    no_poison = deepcopy(authority)
    next(row for row in no_poison if row.get("poisoned"))["poisoned"] = False
    assert "poison_count" in exp.stream_conformance_errors(
        public, no_poison, warmup, manifest, expected_seeds=(exp.STREAM_SEEDS[0],)
    )


def test_nested_public_authority_field_is_rejected() -> None:
    """SCENARIO-CL-7212-STREAM: Nested authority fields also fail leakage checks."""

    assert exp.public_leakage_errors([{"event_id": "bad", "nested": {"hidden_parameter": 2}}])
    assert exp._nested_keys([{"safe": 1}]) == {"safe"}


def test_witness_selector_uses_maximally_balanced_lowest_split() -> None:
    """SCENARIO-CL-7212-WITNESS: Select the deterministic balanced witness."""

    hypotheses = set(range(33))
    chosen = exp.choose_witness("lower_bound", hypotheses, reserved_values={16})
    candidates = []
    for value in exp.PARAMETER_DOMAIN:
        if value == 16:
            continue
        labels = [exp.exact_label("lower_bound", value, parameter) for parameter in hypotheses]
        candidates.append((min(labels.count("accept"), labels.count("reject")), value))
    assert chosen == max(candidates, key=lambda item: (item[0], -item[1]))[1]
    assert exp.choose_witness("lower_bound", {7}, reserved_values=set()) is None
    assert (
        exp.choose_witness("lower_bound", {7, 8}, reserved_values=set(exp.PARAMETER_DOMAIN)) is None
    )


def test_fitting_is_private_and_validation_never_eliminates(tmp_path: Path) -> None:
    """SCENARIO-CL-7212-BUDGET/COMMIT: Validation cannot train the candidate."""

    fallback = exp.FrozenWarmupFallback.from_releases(
        [
            {
                "event_id": f"warm-{index}",
                "family_id": "lower_bound",
                "numeric_value": index,
                "observed_label": "reject",
            }
            for index in range(8)
        ]
    )
    controller = exp.CandidateRefinementController(
        stream_id="test-stream",
        fallback=fallback,
        reserved_validation={"lower_bound": {20, 21, 22, 23}},
    )
    memory = TransactionalConstraintMemory(tmp_path / "memory")
    fit = _public("fit", "lower_bound", 10)
    before = controller.state_dict()["families"]["lower_bound"]["hypotheses"]
    controller.observe_fitting(fit, "accept", memory=memory)
    after_fit = controller.state_dict()["families"]["lower_bound"]["hypotheses"]
    assert after_fit != before

    validation = _public("validation", "lower_bound", 20)
    controller.observe_validation(validation, "accept", memory=memory)
    after_validation = controller.state_dict()["families"]["lower_bound"]["hypotheses"]
    assert after_validation == after_fit
    with pytest.raises(ValueError, match="reserved_validation_used_for_fitting"):
        controller.observe_fitting(validation, "accept", memory=memory)


def test_commit_insertion_and_deletion_change_only_live_memory_path(tmp_path: Path) -> None:
    """SCENARIO-CL-7212-COMMIT/ROLLBACK: Prove the causal deployed seam."""

    receipt = exp.run_commit_path_probe(tmp_path / "probe")
    assert receipt["fallback_prediction"] == "reject"
    assert receipt["committed_prediction"] == "accept"
    assert receipt["deleted_prediction"] == "reject"
    assert receipt["insertion_changed_decision"] is True
    assert receipt["deletion_changed_decision"] is True
    assert receipt["rollback_byte_identical"] is True
    assert receipt["uncommitted_hypotheses_consulted"] is False
    assert receipt["wrong_version_rejected"] is True
    assert receipt["poisoned_transaction_rejected"] is True


def test_controller_rejects_non_validation_and_duplicate_evidence(tmp_path: Path) -> None:
    """SCENARIO-CL-7212-COMMIT: Require distinct reserved validation evidence."""

    fallback = exp.FrozenWarmupFallback.from_releases([])
    controller = exp.CandidateRefinementController(
        stream_id="s",
        fallback=fallback,
        reserved_validation={"lower_bound": {1, 2, 3, 4}},
    )
    memory = TransactionalConstraintMemory(tmp_path / "memory")
    state = controller.families["lower_bound"]
    state.hypotheses = {0}
    state.candidate_parameter = 0
    state.expected_parent_hash = memory.state_hash()
    with pytest.raises(ValueError, match="validation_value_not_reserved"):
        controller.observe_validation(_public("bad", "lower_bound", 7), "accept", memory=memory)
    event = _public("v1", "lower_bound", 1)
    controller.observe_validation(event, "accept", memory=memory)
    with pytest.raises(ValueError, match="duplicate_validation_evidence"):
        controller.observe_validation(event, "accept", memory=memory)


def test_controller_covers_fit_freeze_empty_budget_commit_and_revoke(tmp_path: Path) -> None:
    """SCENARIO-CL-7212-BUDGET/ROLLBACK: Bound and revoke candidate state."""

    fallback = exp.FrozenWarmupFallback.from_releases([])
    reserved = {family: set() for family in exp.FAMILIES}
    reserved["lower_bound"] = {1, 2, 3, 4}
    memory = TransactionalConstraintMemory(tmp_path / "memory")

    freeze = exp.CandidateRefinementController(
        stream_id="freeze", fallback=fallback, reserved_validation=reserved
    )
    freeze.families["lower_bound"].hypotheses = {0, 1}
    receipt = freeze.observe_fitting(
        _public("fit-freeze", "lower_bound", 0), "reject", memory=memory
    )
    assert receipt["operation"] == "freeze_singleton"

    empty = exp.CandidateRefinementController(
        stream_id="empty", fallback=fallback, reserved_validation=reserved
    )
    empty.families["lower_bound"].hypotheses = {0}
    receipt = empty.observe_fitting(_public("fit-empty", "lower_bound", 0), "reject", memory=memory)
    assert receipt["operation"] == "candidate_set_empty"
    empty.fitting_queries = exp.FITTING_BUDGET
    with pytest.raises(ValueError, match="fitting_query_budget_exhausted"):
        empty.observe_fitting(_public("over-fit", "lower_bound", 8), "accept", memory=memory)
    assert empty.commit_candidate("lower_bound", memory=memory)["reason"] == (
        "candidate_not_empirically_validated"
    )

    no_candidate = exp.CandidateRefinementController(
        stream_id="none", fallback=fallback, reserved_validation=reserved
    )
    assert (
        no_candidate.observe_validation(
            _public("no-candidate", "lower_bound", 1), "accept", memory=memory
        )["operation"]
        == "validation_charged_without_candidate"
    )
    no_candidate.validation_queries = exp.VALIDATION_BUDGET
    with pytest.raises(ValueError, match="validation_query_budget_exhausted"):
        no_candidate.observe_validation(
            _public("over-validation", "lower_bound", 2), "accept", memory=memory
        )

    committed_memory = TransactionalConstraintMemory(tmp_path / "committed")
    committed = exp.CandidateRefinementController(
        stream_id="committed", fallback=fallback, reserved_validation=reserved
    )
    state = committed.families["lower_bound"]
    state.hypotheses = {0}
    state.candidate_parameter = 0
    state.expected_parent_hash = committed_memory.state_hash()
    state.validation_ids = ["v1", "v2", "v3"]
    fourth = committed.observe_validation(
        _public("v4", "lower_bound", 4), "accept", memory=committed_memory
    )
    assert fourth["operation"] == "validation_commit"
    contradiction = committed.observe_validation(
        _public("v5", "lower_bound", 1), "reject", memory=committed_memory
    )
    assert contradiction["operation"] == "revoke_contradicted_candidate"
    assert contradiction["rollback_receipt"]["byte_identical"] is True
    assert committed.families["lower_bound"].candidate_parameter is None

    uncommitted = exp.CandidateRefinementController(
        stream_id="uncommitted", fallback=fallback, reserved_validation=reserved
    )
    state = uncommitted.families["lower_bound"]
    state.hypotheses = {0}
    state.candidate_parameter = 0
    state.expected_parent_hash = memory.state_hash()
    contradiction = uncommitted.observe_validation(
        _public("u1", "lower_bound", 1), "reject", memory=memory
    )
    assert contradiction["rollback_receipt"] is None


def test_independent_quarantine_check_uses_metadata_and_manifest() -> None:
    """SCENARIO-CL-7212-PRECONDITIONS: Either quarantine channel blocks use."""

    clean = exp.quarantine_state({}, "", "artifact.json", "exp-clean")
    flagged = exp.quarantine_state({"flagged_adversarial": True}, "", "artifact.json", "exp")
    listed = exp.quarantine_state({}, "- artifact.json\n", "artifact.json", "exp")
    assert clean["quarantined"] is False
    assert flagged["quarantined"] is True
    assert listed["quarantined"] is True


def test_build_and_cold_validate_complete_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7212-TERMINAL: Build a ready fixture without a value claim."""

    paths = exp.ExperimentPaths.under(tmp_path)
    artifact = exp.build_and_seal(exp.REPO_ROOT, paths, duration_s=1.25, progress=True)
    assert artifact["status"] == "complete"
    assert artifact["refinement_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["execution_venue"] == "host"
    assert artifact["execution_host"]
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["upstream_gate_receipt"]["known_failed_value_promoted"] is False
    assert artifact["sample_size_budget"]["independent_units_completed"] == 20
    assert len(artifact["rows"]) == len(exp.STREAM_SEEDS) * len(exp.ARMS)
    assert exp.validate_artifact(artifact, repo_root=exp.REPO_ROOT, check_files=True) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["error"] = 1
    assert "reproducibility_checksum" in exp.validate_artifact(changed)

    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "stream_conformance_errors", lambda *args, **kwargs: ["forced"])
        with pytest.raises(ValueError, match="stream_conformance_failed:forced"):
            exp.build_and_seal(exp.REPO_ROOT, paths, duration_s=1.25)
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
        with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
            exp.build_and_seal(exp.REPO_ROOT, paths, duration_s=1.25)


def test_small_io_and_default_path_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7212-PRECONDITIONS: Invalid input bytes stay explicit."""

    assert exp.ExperimentPaths.defaults().artifact == exp.DEFAULT_ARTIFACT_PATH
    derived = exp.ExperimentPaths.from_stream_root(tmp_path / "stream")
    assert derived.checkpoint.name == "experiment_7212.json"
    assert derived.artifact.name == "experiment_7212.json"
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_row_not_object"):
        exp.read_jsonl(bad_jsonl)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._read_summary(malformed) == {}
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError()),
    )
    assert exp._read_summary(malformed) == {}
    assert exp._sha256_path(tmp_path / "missing") is None


def test_missing_upstream_builds_diagnostic_row_free_block(tmp_path: Path) -> None:
    """SCENARIO-CL-7212-PRECONDITIONS: External absence is terminal blocked."""

    paths = exp.ExperimentPaths.under(tmp_path / "out")
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        upstream_7199=tmp_path / "missing-7199.json",
        upstream_7200=tmp_path / "missing-7200.json",
        duration_s=0.5,
        progress=True,
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    summary = artifact["gate_check_summary"]
    assert summary["passed"] is False
    assert all(summary[key] is not None for key in ("failed_check", "upstream", "field"))
    assert exp.validate_artifact(artifact) == []


def test_cli_main_writes_valid_terminal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7212: The executable path performs a final atomic write."""

    expected = exp.build_blocked_artifact(
        [exp.gate_check("forced", "test", "value", True, False)],
        source_hashes={},
        paths=exp.ExperimentPaths.under(tmp_path),
        duration_s=0.1,
    )
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: expected)
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)]) == 0
    written = json.loads(exp.ExperimentPaths.under(tmp_path).artifact.read_text(encoding="utf-8"))
    assert written == expected


def test_cli_worker_and_validation_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7212: Worker inputs and final validation both fail closed."""

    with pytest.raises(ValueError, match="evaluator_worker_requires_stream_root"):
        exp.main(["--evaluator-worker"])
    monkeypatch.setattr(exp, "evaluator_worker", lambda paths: 7)
    assert exp.main(["--evaluator-worker", "--stream-root", str(tmp_path)]) == 7

    artifact = exp.build_blocked_artifact(
        [exp.gate_check("forced", "test", "value", True, False)],
        source_hashes={},
        paths=exp.ExperimentPaths.under(tmp_path),
        duration_s=0.1,
    )
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp.main(["--output-root", str(tmp_path)])


def test_validate_missing_fields() -> None:
    """SCENARIO-CL-7212-TERMINAL: An incomplete artifact never validates."""

    assert exp.validate_artifact({})[0].startswith("missing_fields:")


def test_cli_rejects_wrong_date() -> None:
    """SCENARIO-CL-7212-PRECONDITIONS: Do not substitute another run date."""

    with pytest.raises(ValueError, match="run_date_must_equal_20260911"):
        exp.main(["--date", "20260910"])
