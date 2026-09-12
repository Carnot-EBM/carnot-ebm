"""Focused tests for bounded coverage memory and effective controls.

Spec refs: REQ-CL-7253 and SCENARIO-CL-7253-*.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7253_v638_coverage_memory as exp


def _release(event_id: str, family: str, value: int, label: str, index: int) -> dict[str, object]:
    """Build released-only evidence for the transaction scenarios."""

    return {
        "event_id": event_id,
        "family_id": family,
        "numeric_value": value,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def _singleton(parameter: int = 0) -> exp7226.PackedBeliefController:
    """Create a finite state with one hypothesis in every family."""

    return exp7226.PackedBeliefController.from_survivors(
        {family: {parameter} for family in exp7226.FAMILIES}
    )


@pytest.fixture(scope="module")
def prospective_views() -> exp.StreamViews:
    """Generate the frozen prospective bytes once for focused replay tests."""

    return exp.build_stream_views("prospective")


@pytest.fixture(scope="module")
def one_stream_panel(prospective_views: exp.StreamViews) -> exp.FixturePanel:
    """Replay all eight arms on one full stream once."""

    return exp.run_fixture_panel(
        prospective_views,
        stream_ids=("prospective-01",),
        progress=True,
    )


@pytest.fixture(scope="module")
def one_stream_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, dict[str, object]]:
    """Build one sidecar-backed terminal object once for cold-validator tests."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7253-artifact"))
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        progress=True,
    )
    return paths, artifact


def test_req_cl_7253_spec_and_frozen_contract() -> None:
    """REQ-CL-7253 fixes counts, arms, zero-model work, and field principles."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7253" in spec
    assert len(set(exp.SCENARIO_PATTERN.findall(spec))) == 8
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert len(exp.ARMS) == 8
    assert len(exp.ARCHIVE_ARMS) == 4
    assert exp.STREAM_COUNT == 32
    assert exp.DEVELOPMENT_STREAM_COUNT == 8
    assert exp.EVENTS_PER_STREAM == 1_024
    assert exp.WARMUP_COUNT == 128
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)
    assert exp.ExperimentPaths.defaults().artifact == exp.REPO_ROOT / exp.DEFAULT_ARTIFACT


def test_scenario_cl_7253_coverage_deduplicates_and_differs_from_fifo() -> None:
    """SCENARIO-CL-7253-COVERAGE checks duplicate removal and farthest-first retention."""

    signatures = ("aaaa", "aaaa", "aaar", "aarr", "arrr", "rrrr")
    archives = [
        exp.diagnostic_archive(f"archive-{index}", index, tuple(signature))
        for index, signature in enumerate(signatures)
    ]
    fifo, fifo_receipt = exp.retain_archives(archives, mode="fifo", archive_cap=4)
    coverage, coverage_receipt = exp.retain_archives(
        archives,
        mode="coverage",
        archive_cap=4,
    )

    assert [row["archive_id"] for row in fifo] == [
        "archive-2",
        "archive-3",
        "archive-4",
        "archive-5",
    ]
    assert len({tuple(row["signature"]) for row in coverage}) == 4
    assert [row["archive_id"] for row in coverage] != [row["archive_id"] for row in fifo]
    assert fifo_receipt["duplicate_eliminated_count"] == 1
    assert coverage_receipt["distance"] == "hamming_over_released_witness_signature"
    assert coverage_receipt["tie_break"] == ["newer_creation_order", "archive_id"]


def test_scenario_cl_7253_shuffle_changes_eligible_identity() -> None:
    """SCENARIO-CL-7253-SHUFFLE proves a candidate-identity intervention."""

    receipt = exp.run_shuffle_diagnostic()
    assert receipt["candidate_count"] == 2
    assert receipt["distinguishable_candidate_count"] == 2
    assert receipt["before_mapping"] != receipt["after_mapping"]
    assert receipt["selected_before_archive_id"] is not None
    assert receipt["selected_after_archive_id"] is not None
    assert receipt["selected_before_archive_id"] != receipt["selected_after_archive_id"]
    assert receipt["selection_changed"] is True
    assert receipt["same_archive_contents"] is True
    assert receipt["same_final_safety_gate"] is True


def test_scenario_cl_7253_transaction_is_future_only_and_atomic(tmp_path: Path) -> None:
    """SCENARIO-CL-7253-TRANSACTION checks delayed commit, rejection, reload, and rollback."""

    controller = exp.CoverageArchiveController.from_active(_singleton())
    probe = {"event_id": "probe", "family_id": "lower_bound", "numeric_value": 0}
    before_prediction = controller.predict(probe)
    before_bytes = controller.state_bytes()
    state_path = tmp_path / "controller.json"
    controller.save(state_path)

    with pytest.raises(exp.ArchiveCommitRejected, match="stale_parent"):
        controller.commit_batch(
            [_release("wrong-parent", "lower_bound", 0, "reject", 1)],
            current_cycle=1,
            expected_parent_hash="sha256:" + "0" * 64,
            state_path=state_path,
        )
    assert controller.state_bytes() == before_bytes == state_path.read_bytes()

    future = _release("future", "lower_bound", 0, "reject", 2)
    future["release_index"] = 3
    with pytest.raises(exp.ArchiveCommitRejected, match="future_release"):
        controller.commit_batch(
            [future],
            current_cycle=2,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    assert controller.state_bytes() == before_bytes == state_path.read_bytes()

    receipt = controller.commit_batch(
        [_release("delayed", "lower_bound", 0, "reject", 4)],
        current_cycle=4,
        expected_parent_hash=controller.state_hash(),
        state_path=state_path,
    )
    assert receipt["operations"][0]["same_event_correction"] is False
    assert before_prediction[0] == "accept"
    assert controller.predict(probe)[0] == "reject"
    assert exp.CoverageArchiveController.load(state_path).state_bytes() == controller.state_bytes()

    rollback = controller.rollback(receipt, state_path=state_path)
    assert rollback["byte_identical"] is True
    assert controller.state_bytes() == before_bytes == state_path.read_bytes()
    with pytest.raises(exp.ArchiveCommitRejected, match="stale_rollback"):
        controller.rollback(receipt)


def test_scenario_cl_7253_memory_caps_every_mutable_collection() -> None:
    """SCENARIO-CL-7253-MEMORY fills the bounded witness, archive, pending, and ledger state."""

    controller = exp.CoverageArchiveController.from_active(_singleton())
    for index in range(48):
        label = "reject" if index % 2 == 0 else "accept"
        controller.commit_batch(
            [_release(f"memory-{index}", "lower_bound", 0, label, index)],
            current_cycle=index,
            expected_parent_hash=controller.state_hash(),
        )
    pending = [
        {"event_id": f"pending-{index}", "release_index": 100 + index}
        for index in range(exp.PENDING_CAPACITY)
    ]
    usage = controller.memory_usage(pending)
    state = controller.state_dict()

    assert len(state["release_window"]) == exp.VALIDATION_WINDOW
    assert len(state["ledger"]) == exp.LEDGER_CAPACITY
    assert len(state["archives"]) <= exp.ARCHIVE_CAP
    assert usage["pending_count"] == exp.PENDING_CAPACITY
    assert usage["within_all_caps"] is True
    assert all(
        usage[f"{component}_bytes"] <= exp.MEMORY_CAPS[f"{component}_bytes"]
        for component in ("witness", "archive", "pending", "ledger", "controller")
    )
    assert all(not family["provenance"] for family in state["active"]["families"].values())


def test_scenario_cl_7253_state_validation_fails_closed() -> None:
    """SCENARIO-CL-7253-MEMORY rejects malformed bounded state without admission."""

    controller = exp.CoverageArchiveController()
    cases = []
    for field, value in (
        ("schema", "wrong"),
        ("archive_cap", True),
        ("archive_cap", 5),
        ("admission_mode", "wrong"),
        ("nomination_mode", "wrong"),
        ("release_window", [{}] * 17),
        ("ledger", [str(index) for index in range(17)]),
        ("archives", [{}] * 5),
    ):
        changed = controller.state_dict()
        changed[field] = value
        cases.append(changed)
    duplicate_order = controller.state_dict()
    archive = exp.diagnostic_archive("a", 0, ("accept",))
    duplicate_order["archives"] = [archive, {**archive, "archive_id": "b"}]
    cases.append(duplicate_order)
    for changed in cases:
        with pytest.raises(ValueError):
            exp.CoverageArchiveController.from_state(changed)


def test_scenario_cl_7253_constructor_and_archive_validation_branches(tmp_path: Path) -> None:
    """SCENARIO-CL-7253-MEMORY rejects each invalid constructor and archive shape."""

    for kwargs in (
        {"archive_cap": 5},
        {"admission_mode": "invalid"},
        {"nomination_mode": "invalid"},
    ):
        with pytest.raises(ValueError):
            exp.CoverageArchiveController(**kwargs)
    for mode, cap in (("invalid", 4), ("fifo", 5)):
        with pytest.raises(ValueError):
            exp.retain_archives([], mode=mode, archive_cap=cap)

    controller = exp.CoverageArchiveController()
    base = controller.state_dict()
    active_history = deepcopy(base)
    active_history["active"]["families"][exp.FAMILIES[0]]["provenance"] = [{}]
    malformed_archive = deepcopy(base)
    malformed_archive["archives"] = ["not-an-object"]
    invalid_masks = deepcopy(base)
    invalid_masks["archives"] = [exp.diagnostic_archive("a", 0, ("accept",))]
    invalid_masks["archives"][0]["survivor_masks"].pop(exp.FAMILIES[0])
    invalid_hash = deepcopy(base)
    invalid_hash["archives"] = [exp.diagnostic_archive("a", 0, ("accept",))]
    invalid_hash["archives"][0]["state_hash"] = "sha256:bad"
    invalid_signature = deepcopy(base)
    invalid_signature["archives"] = [exp.diagnostic_archive("a", 0, ("accept",))]
    invalid_signature["archives"][0]["signature"] = ["unknown"]
    duplicate_signature = deepcopy(base)
    duplicate_signature["archives"] = [
        exp.diagnostic_archive("a", 0, ("accept",)),
        exp.diagnostic_archive("b", 1, ("accept",)),
    ]
    oversized = deepcopy(base)
    oversized["last_nomination_receipt"] = {"padding": "x" * exp.MEMORY_CAPS["controller_bytes"]}
    for changed in (
        active_history,
        malformed_archive,
        invalid_masks,
        invalid_hash,
        invalid_signature,
        duplicate_signature,
        oversized,
    ):
        with pytest.raises(ValueError):
            exp.CoverageArchiveController.from_state(changed)

    non_object = tmp_path / "state.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_archive_state_object"):
        exp.CoverageArchiveController.load(non_object)
    assert controller.archives() == []
    block = [{"event_id": "q", "family_id": "lower_bound", "numeric_value": 0}]
    assert controller.select_request(block, {"q": 0}) == block[0]


def test_req_cl_7253_io_and_malformed_raw_rows(tmp_path: Path) -> None:
    """REQ-CL-7253 rejects changed seals and unavailable or malformed raw evidence."""

    sealed = tmp_path / "sealed.json"
    first = exp._write_immutable(sealed, b"same")
    second = exp._write_immutable(sealed, b"same")
    assert first == second
    with pytest.raises(exp.exp7240.ImmutableSealError):
        exp._write_immutable(sealed, b"changed")
    missing = tmp_path / "missing.jsonl"
    assert exp._read_jsonl(missing) == []
    missing.write_text("not-json\n", encoding="utf-8")
    assert exp._read_jsonl(missing) == []
    missing.write_text("[]\n", encoding="utf-8")
    assert exp._read_jsonl(missing) == []
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp.independent_reduce(missing)
    bad_row = {
        "prediction_frozen_before_release": True,
        "oracle_control": False,
        "held_out_label_visible_to_controller": True,
        "controller_input_fields": [],
    }
    missing.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_row_validation_failed:future_label_leakage"):
        exp.independent_reduce(missing)
    assert exp._nested_keys([{"inside": 1}, "scalar"]) == {"inside"}
    with pytest.raises(ValueError, match="invalid_stream_kind"):
        exp.build_stream_views("invalid")


def test_scenario_cl_7253_streams_are_new_separated_and_conform(
    prospective_views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7253-STREAMS checks new seeds, schedule families, noise, and isolation."""

    development = exp.build_stream_views("development")
    assert exp.stream_conformance_errors(development, "development") == []
    assert exp.stream_conformance_errors(prospective_views, "prospective") == []
    assert len(development.public) == exp.DEVELOPMENT_STREAM_COUNT * exp.EVENTS_PER_STREAM
    assert len(prospective_views.public) == exp.STREAM_COUNT * exp.EVENTS_PER_STREAM
    assert not (
        {row["stream_seed"] for row in development.authority}
        & {row["stream_seed"] for row in prospective_views.authority}
    )
    assert exp.public_leakage_errors(prospective_views.public) == []
    assert any(row["release_noisy"] for row in prospective_views.releases)
    assert set(row["delay"] for row in prospective_views.releases) == set(exp.DELAY_SUPPORT)
    assert prospective_views.manifest["patterns"] == dict.fromkeys(exp.PATTERNS, 8)

    leaked = deepcopy(prospective_views.public[:1])
    leaked[0]["regime_id"] = "private"
    assert exp.public_leakage_errors(leaked) == [leaked[0]["event_id"]]
    changed = deepcopy(prospective_views)
    changed.public.pop()
    assert "event_count" in exp.stream_conformance_errors(changed, "prospective")

    identity = deepcopy(prospective_views)
    identity.releases[0]["event_id"] = "wrong"
    assert "event_identity" in exp.stream_conformance_errors(identity, "prospective")
    leaked_views = deepcopy(prospective_views)
    leaked_views.public[0]["exact_label"] = "accept"
    assert "public_authority_leakage" in exp.stream_conformance_errors(leaked_views, "prospective")
    patterns = deepcopy(prospective_views)
    patterns.manifest["patterns"] = {}
    assert "pattern_balance" in exp.stream_conformance_errors(patterns, "prospective")
    chronology = deepcopy(prospective_views)
    chronology.public[0]["chronology_index"] = 99
    assert "chronology" in exp.stream_conformance_errors(chronology, "prospective")
    recurrence = deepcopy(prospective_views)
    recurrence.public[768]["numeric_value"] += 1
    assert "identical_input_recurrence" in exp.stream_conformance_errors(recurrence, "prospective")
    grounding = deepcopy(prospective_views)
    grounding.releases[0]["observed_label"] = (
        "reject" if grounding.releases[0]["observed_label"] == "accept" else "accept"
    )
    assert "authority_grounding" in exp.stream_conformance_errors(grounding, "prospective")


def test_scenario_cl_7253_panel_has_eight_arms_and_independent_reduction(
    one_stream_panel: exp.FixturePanel,
) -> None:
    """SCENARIO-CL-7253-ARMS checks complete rows, shared budgets, and raw reduction."""

    panel = one_stream_panel
    assert len(panel.event_rows) == exp.EVENTS_PER_STREAM * len(exp.ARMS)
    assert len(panel.rows) == len(exp.ARMS)
    assert {row["arm"] for row in panel.rows} == set(exp.ARMS)
    assert exp.panel_conformance_errors(panel, ("prospective-01",)) == []
    reduced = exp.reduce_event_rows(panel.event_rows)
    assert reduced == panel.rows
    archive_rows = [row for row in panel.rows if row["arm"] in exp.ARCHIVE_ARMS]
    assert len({row["query_count"] for row in archive_rows}) == 1
    assert len({row["release_count"] for row in archive_rows}) == 1
    assert all(
        row["maximum_memory_bytes"] <= exp.MEMORY_CAPS["total_bytes"] for row in archive_rows
    )

    changed = deepcopy(panel.event_rows)
    changed[0]["prediction_frozen_before_release"] = False
    assert "prediction_chronology" in exp.event_row_errors(changed)

    short = replace(panel, event_rows=panel.event_rows[:-1])
    assert "event_row_count" in exp.panel_conformance_errors(short, ("prospective-01",))
    missing_unit = replace(panel, rows=panel.rows[:-1])
    assert "unit_rows" in exp.panel_conformance_errors(missing_unit, ("prospective-01",))
    mismatched_rows = deepcopy(panel.rows)
    archive_row = next(row for row in mismatched_rows if row["arm"] in exp.ARCHIVE_ARMS)
    archive_row["query_count"] += 1
    mismatch = replace(panel, rows=mismatched_rows)
    assert "archive_budget_mismatch" in exp.panel_conformance_errors(mismatch, ("prospective-01",))
    maximum = dict(panel.maximum_memory)
    maximum["total_bytes"] = exp.MEMORY_CAPS["total_bytes"] + 1
    over = replace(panel, maximum_memory=maximum)
    assert "memory_cap" in exp.panel_conformance_errors(over, ("prospective-01",))


def test_scenario_cl_7253_controls_cover_e2e_and_effective_admission(tmp_path: Path) -> None:
    """SCENARIO-CL-7253-TRANSACTION runs E2E-007 and diagnostic positive controls."""

    rows = exp.run_controller_controls(tmp_path)
    assert all(row["passed"] is True for row in rows)
    assert {row["control"] for row in rows} == {
        "bounded_memory",
        "failed_parent_atomic_rollback",
        "delayed_feedback_future_only",
        "future_label_rejected",
        "coverage_eviction_differs_from_fifo",
        "effective_signature_shuffle",
        "cold_reload_and_rollback",
    }


def test_scenario_cl_7253_preconditions_and_blocked_terminal(tmp_path: Path) -> None:
    """SCENARIO-CL-7253-PRECONDITIONS authenticates inputs or emits a row-free block."""

    paths = exp.ExperimentPaths.under(tmp_path / "valid")
    checks, hashes, upstream = exp.collect_preconditions(exp.REPO_ROOT, paths)
    assert exp.gate_summary(checks)["passed"] is True
    assert set(upstream) == {"exp7240", "exp7241", "exp7242"}
    assert all(hashes[str(exp.REPO_ROOT / path)] for path in exp.UPSTREAM_ARTIFACTS.values())

    absent = tmp_path / "absent.json"
    blocked_paths = exp.ExperimentPaths.under(tmp_path / "blocked")
    blocked = exp.build_and_seal(
        exp.REPO_ROOT,
        blocked_paths,
        upstream_overrides={"exp7240": absent},
        progress=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["coverage_fixture_ready_score"] == 0
    assert blocked["inference_substrate"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["failed_check"] == "exp7240_artifact_hash"
    assert exp.validate_artifact(blocked, repo_root=exp.REPO_ROOT, check_files=False) == []


def test_scenario_cl_7253_one_stream_artifact_and_cold_validation(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7253-TERMINAL builds, reduces, validates, and seals a scoped fixture."""

    paths, artifact = one_stream_artifact
    assert artifact["status"] == "complete"
    assert artifact["coverage_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["science_learning_gain_scored"] is False
    assert len(artifact["rows"]) == len(exp.ARMS)
    assert artifact["shuffle_effect_receipt"]["selection_changed"] is True
    assert artifact["sample_size_budget"]["independent_units_completed"] == 1
    assert exp.independent_reduce(paths.raw_rows) == artifact["rows"]
    assert (
        exp.validate_artifact(
            artifact,
            repo_root=exp.REPO_ROOT,
            expected_stream_ids=("prospective-01",),
            check_files=True,
        )
        == []
    )

    exp.write_artifact(
        paths.artifact,
        artifact,
        repo_root=exp.REPO_ROOT,
        expected_stream_ids=("prospective-01",),
    )
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact


def test_req_cl_7253_corrupt_commit_and_rollback_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7253 covers corrupt live, durable, duplicate, and rollback receipts."""

    controller = exp.CoverageArchiveController.from_active(_singleton())
    controller._state["schema"] = "corrupt"
    with pytest.raises(exp.ArchiveCommitRejected, match="corrupt_live_state"):
        controller.commit_batch(
            [_release("corrupt-live", "lower_bound", 0, "accept", 0)],
            current_cycle=0,
            expected_parent_hash=controller.state_hash(),
        )

    controller = exp.CoverageArchiveController.from_active(_singleton())
    state_path = tmp_path / "state.json"
    state_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(exp.ArchiveCommitRejected, match="corrupt_durable_state"):
        controller.commit_batch(
            [_release("corrupt-durable", "lower_bound", 0, "accept", 0)],
            current_cycle=0,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    exp.CoverageArchiveController().save(state_path)
    with pytest.raises(exp.ArchiveCommitRejected, match="stale_durable_parent"):
        controller.commit_batch(
            [_release("stale-durable", "lower_bound", 0, "accept", 0)],
            current_cycle=0,
            expected_parent_hash=controller.state_hash(),
            state_path=state_path,
        )
    duplicate = _release("duplicate", "lower_bound", 0, "accept", 0)
    with pytest.raises(exp.ArchiveCommitRejected, match="duplicate_release"):
        controller.commit_batch(
            [duplicate, duplicate],
            current_cycle=0,
            expected_parent_hash=controller.state_hash(),
        )

    receipt = controller.commit_batch(
        [_release("valid", "lower_bound", 0, "reject", 1)],
        current_cycle=1,
        expected_parent_hash=controller.state_hash(),
    )
    invalid = dict(receipt, parent_bytes_b64="not-base64")
    with pytest.raises(exp.ArchiveCommitRejected, match="invalid_rollback_receipt"):
        controller.rollback(invalid)
    wrong_parent = dict(receipt, parent_hash="sha256:" + "0" * 64)
    with pytest.raises(exp.ArchiveCommitRejected, match="rollback_parent_hash"):
        controller.rollback(wrong_parent)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "wrong", "identity"),
        ("MODEL_SPECS", [{"model": "forbidden"}], "model_invocation"),
        ("execution_venue", "gpu", "execution_venue"),
        ("verifier_is_oracle", False, "oracle_declaration"),
        ("coverage_fixture_ready_score", 1, "blocked_contract"),
        ("validation_receipts", [{"command": "bad"}], "validation_receipts"),
    ],
)
def test_req_cl_7253_cold_validator_rejects_terminal_drift(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    """REQ-CL-7253 rejects changed identity, invocation, oracle, readiness, and receipts."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks = [exp.gate_check("forced", "test", "ready", True, False)]
    artifact = exp.build_blocked_artifact(checks, {}, {}, paths)
    changed = deepcopy(artifact)
    changed[field] = value
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert error in exp.validate_artifact(changed, repo_root=exp.REPO_ROOT, check_files=False)


def test_req_cl_7253_complete_validator_and_write_failure_branches(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
    tmp_path: Path,
) -> None:
    """REQ-CL-7253 rejects non-object, nonterminal, bad complete fields, and bad writes."""

    paths, artifact = one_stream_artifact
    assert exp.validate_artifact([], check_files=False) == ["artifact_mapping"]
    running = deepcopy(artifact)
    running["status"] = "running"
    running["reproducibility_checksum"] = exp.reproducibility_checksum(running)
    assert "status" in exp.validate_artifact(running, check_files=False)
    mutations = (
        ({"inference_substrate": "wrong"}, "substrate"),
        ({"gate_check_summary": {"passed": False}}, "preconditions"),
        ({"rows": []}, "rows"),
        ({"acceptance_gate_results": {}}, "acceptance_gate_results"),
        ({"shuffle_effect_receipt": {}}, "shuffle_effect_receipt"),
        ({"controller_contract": {}}, "controller_contract"),
        ({"sample_size_budget": {}}, "sample_size_budget"),
    )
    for values, expected in mutations:
        changed = deepcopy(artifact)
        changed.update(values)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(
            changed,
            expected_stream_ids=("prospective-01",),
            check_files=False,
        )

    changed = deepcopy(artifact)
    changed["rows"] = []
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(
            tmp_path / "invalid.json",
            changed,
            expected_stream_ids=("prospective-01",),
        )
    assert paths.raw_rows.is_file()


def test_req_cl_7253_validation_receipts_and_thin_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7253 records observed validation and keeps command dispatch thin."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks = [exp.gate_check("forced", "test", "ready", True, False)]
    artifact = exp.build_blocked_artifact(checks, {}, {}, paths)
    receipt = {
        "command": "pytest focused",
        "exit_code": 0,
        "classification": "passed",
        "log_sha256": "sha256:" + "0" * 64,
    }
    attached = exp.attach_validation_receipts(artifact, [receipt])
    assert attached["validation_receipts"] == [receipt]
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(artifact, [{"command": "incomplete"}])

    called: list[object] = []
    monkeypatch.setattr(exp, "main", lambda argv=None: called.append(argv) or 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(exp.REPO_ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert raised.value.code == 0
    assert called == [None]


def test_req_cl_7253_main_date_and_validation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7253 rejects the wrong date and fails before an invalid terminal write."""

    with pytest.raises(SystemExit, match="run_date_must_be_20260912"):
        exp.main(["--date", "20260911"])
    artifact = exp.build_blocked_artifact(
        [exp.gate_check("forced", "test", "ready", True, False)],
        {},
        {},
        exp.ExperimentPaths.defaults(),
    )
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp.main(["--date", exp.RUN_DATE])


def test_req_cl_7253_main_success_path(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7253 prints validation and write boundaries on a valid CLI run."""

    _, artifact = one_stream_artifact
    writes: list[Path] = []
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        exp,
        "write_artifact",
        lambda path, *args, **kwargs: writes.append(path),
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert writes == [exp.ExperimentPaths.defaults().artifact]


@pytest.mark.parametrize(
    "failure",
    ["stream", "panel", "controls", "reducer", "artifact"],
)
def test_req_cl_7253_build_fails_closed_at_each_internal_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    one_stream_panel: exp.FixturePanel,
    failure: str,
) -> None:
    """REQ-CL-7253 prevents terminal publication after any internal gate fails."""

    paths = exp.ExperimentPaths.under(tmp_path / failure)
    if failure == "stream":
        monkeypatch.setattr(exp, "stream_conformance_errors", lambda *args: ["forced"])
        expected = "stream_conformance_failed"
    else:
        monkeypatch.setattr(exp, "run_fixture_panel", lambda *args, **kwargs: one_stream_panel)
        if failure == "panel":
            monkeypatch.setattr(exp, "panel_conformance_errors", lambda *args: ["forced"])
            expected = "panel_conformance_failed"
        elif failure == "controls":
            monkeypatch.setattr(
                exp,
                "run_controller_controls",
                lambda *args: [{"control": "forced", "passed": False}],
            )
            expected = "controller_control_failed"
        elif failure == "reducer":
            monkeypatch.setattr(exp, "independent_reduce", lambda *args: [])
            expected = "independent_reducer_mismatch"
        else:
            real_validate = exp.validate_artifact
            calls = 0

            def fail_terminal(*args: object, **kwargs: object) -> list[str]:
                nonlocal calls
                calls += 1
                return ["forced"] if calls else real_validate(*args, **kwargs)

            monkeypatch.setattr(exp, "validate_artifact", fail_terminal)
            expected = "artifact_validation_failed"
    with pytest.raises(ValueError, match=expected):
        exp.build_and_seal(
            exp.REPO_ROOT,
            paths,
            stream_ids=("prospective-01",),
            progress=False,
        )
