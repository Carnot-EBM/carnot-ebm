"""Tests for the bounded pending-feedback constraint stream.

Spec refs: REQ-CL-7198 and SCENARIO-CL-7198-*.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_7198_v634_feedback_capacity_stream as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def views() -> exp.StreamViews:
    """Build all frozen rows once because every test treats them as immutable."""

    return exp.build_stream_views()


@pytest.fixture(scope="module")
def panel(views: exp.StreamViews) -> exp.CapacityPanel:
    """Run the bounded scheduler once so tests can inspect the same replay."""

    return exp.run_capacity_panel(views)


@pytest.fixture()
def artifact(tmp_path: Path) -> dict[str, object]:
    """Seal private files so tests never overwrite the research deliverable."""

    return exp.build_and_seal(
        REPO_ROOT,
        exp.ExperimentPaths.under(tmp_path),
        run_date=exp.RUN_DATE,
        duration_s=1.25,
    )


def test_fixed_contract_and_principles_are_complete() -> None:
    """REQ-CL-7198: Seeds, windows, cells, and field reasons are frozen."""

    assert len(exp.STREAM_SEEDS) == 10
    assert len(set(exp.STREAM_SEEDS)) == 10
    assert exp.EVENTS_PER_SEED == 1024
    assert sum(window["count"] for window in exp.WINDOWS) == 1024
    assert [window["count"] for window in exp.WINDOWS] == [128, 128, 512, 128, 128]
    assert exp.PARAMETER_DOMAIN == tuple(range(33))
    assert len(exp.FAMILIES) == 4
    assert exp.CAPACITIES == (1, 4, 16)
    assert exp.DELAY_SCHEDULES == ("constant_0", "constant_4", "constant_16", "burst")
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.EXECUTION_VENUE == "host"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)


def test_stream_is_frozen_balanced_and_authority_is_separate(views: exp.StreamViews) -> None:
    """SCENARIO-CL-7198-STREAM: All windows balance families without public truth."""

    assert len(views.public_events) == 10_240
    assert len(views.authority_events) == 10_240
    assert exp.stream_conformance_errors(views) == []
    assert exp.public_leakage_errors(views.public_events) == []
    assert len({row["event_id"] for row in views.public_events}) == 10_240
    poisoned = [row for row in views.authority_events if row["poisoned"]]
    assert len(poisoned) == 160
    assert all(row["observed_label"] != row["exact_label"] for row in poisoned)
    assert all(
        row["observed_label"] == row["exact_label"]
        for row in views.authority_events
        if not row["poisoned"]
    )
    for seed in exp.STREAM_SEEDS:
        authority = [row for row in views.authority_events if row["seed"] == seed]
        for window in exp.WINDOWS:
            rows = [row for row in authority if row["window"] == window["name"]]
            assert len(rows) == window["count"]
            assert {
                family: sum(row["family_id"] == family for row in rows) for family in exp.FAMILIES
            } == {family: window["count"] // 4 for family in exp.FAMILIES}


def test_public_parser_execution_and_independent_scoring_agree(views: exp.StreamViews) -> None:
    """SCENARIO-CL-7198-GROUNDING: Public text reaches two agreeing exact scorers."""

    public_by_id = {row["event_id"]: row for row in views.public_events}
    for truth in views.authority_events:
        public = public_by_id[truth["event_id"]]
        extracted = exp.extract_public_input(public["public_input"])
        assert extracted == {
            "family_id": public["family_id"],
            "numeric_value": public["numeric_value"],
        }
        assert (
            exp.exact_label(**extracted, parameter=truth["hidden_parameter"])
            == truth["exact_label"]
        )
        assert (
            exp.independent_exact_label(**extracted, parameter=truth["hidden_parameter"])
            == truth["independent_exact_label"]
        )
        assert truth["exact_label"] == truth["independent_exact_label"]


def test_public_and_feedback_iterators_preserve_chronology(views: exp.StreamViews) -> None:
    """SCENARIO-CL-7198-CHRONOLOGY: Only the release iterator yields a label."""

    first_public = next(exp.public_event_iterator(views.public_events))
    assert exp._nested_keys(first_public).isdisjoint(exp.FORBIDDEN_PUBLIC_FIELDS)
    pending = [
        exp.PendingRecord(
            event_id="fixture-event",
            family_id="lower_bound",
            numeric_value=7,
            request_index=3,
            release_index=7,
            observed_label="accept",
            exact_label="accept",
            poisoned=False,
        )
    ]
    assert list(exp.feedback_release_iterator(pending, 6)) == []
    released = list(exp.feedback_release_iterator(pending, 7))
    assert released == [
        {
            "event_id": "fixture-event",
            "family_id": "lower_bound",
            "numeric_value": 7,
            "request_index": 3,
            "release_index": 7,
            "observed_label": "accept",
            "exact_label": "accept",
            "poisoned": False,
        }
    ]


def test_public_api_rejects_malformed_and_authority_inputs() -> None:
    """SCENARIO-CL-7198-CHRONOLOGY: Defensive public APIs fail closed."""

    assert exp.ExperimentPaths.defaults().artifact == exp.DEFAULT_ARTIFACT_PATH
    assert exp._nested_keys([{"nested": {"label": "accept"}}]) == {
        "nested",
        "label",
    }
    with pytest.raises(ValueError, match="invalid_public_input"):
        exp.extract_public_input("family=lower_bound;hidden=3")
    with pytest.raises(ValueError, match="unknown_family:unknown"):
        exp.exact_label("unknown", 1, 1)
    with pytest.raises(ValueError, match="unknown_family:unknown"):
        exp.independent_exact_label("unknown", 1, 1)
    with pytest.raises(ValueError, match="public_event_contains_authority_field"):
        next(exp.public_event_iterator([{"event_id": "bad", "exact_label": "accept"}]))
    with pytest.raises(ValueError, match="unsupported_admission_arm"):
        exp.select_request(
            [{"event_id": "bad", "family_id": "lower_bound", "numeric_value": 1}],
            "static_frozen",
            {"bad": 0},
        )


def test_stream_conformance_names_each_isolated_mutation(views: exp.StreamViews) -> None:
    """SCENARIO-CL-7198-MUTATIONS: Every stream failure has a stable detector."""

    assert "public_event_count" in exp.stream_conformance_errors(
        replace(views, public_events=views.public_events[:-1])
    )
    assert "authority_event_count" in exp.stream_conformance_errors(
        replace(views, authority_events=views.authority_events[:-1])
    )

    public = deepcopy(views.public_events)
    public[0]["event_id"] = "changed"
    assert "event_identity" in exp.stream_conformance_errors(replace(views, public_events=public))

    public = deepcopy(views.public_events)
    public[0]["exact_label"] = "accept"
    assert "public_access_leakage" in exp.stream_conformance_errors(
        replace(views, public_events=public)
    )

    manifest = deepcopy(views.manifest)
    manifest["windows"][1]["start"] = 127
    assert "window_partition" in exp.stream_conformance_errors(replace(views, manifest=manifest))

    public = deepcopy(views.public_events)
    public[0]["numeric_value"] += 1
    assert "public_authority_join" in exp.stream_conformance_errors(
        replace(views, public_events=public)
    )

    public = deepcopy(views.public_events)
    public[0]["public_input"] = "malformed"
    assert "public_extraction" in exp.stream_conformance_errors(
        replace(views, public_events=public)
    )

    public = deepcopy(views.public_events)
    public[0]["public_input"] = f"family={public[0]['family_id']};value=-1"
    assert "public_extraction" in exp.stream_conformance_errors(
        replace(views, public_events=public)
    )

    authority = deepcopy(views.authority_events)
    authority[0]["independent_exact_label"] = (
        "reject" if authority[0]["exact_label"] == "accept" else "accept"
    )
    assert "exact_label_disagreement" in exp.stream_conformance_errors(
        replace(views, authority_events=authority)
    )

    authority = deepcopy(views.authority_events)
    authority[0]["observed_label"] = (
        "reject" if authority[0]["exact_label"] == "accept" else "accept"
    )
    assert "poison_witness" in exp.stream_conformance_errors(
        replace(views, authority_events=authority)
    )

    public = deepcopy(views.public_events)
    authority = deepcopy(views.authority_events)
    replacement_family = next(family for family in exp.FAMILIES if family != public[0]["family_id"])
    public[0]["family_id"] = replacement_family
    public[0]["public_input"] = f"family={replacement_family};value={public[0]['numeric_value']}"
    authority[0]["family_id"] = replacement_family
    assert "family_balance" in exp.stream_conformance_errors(
        replace(views, public_events=public, authority_events=authority)
    )


def test_capacity_panel_matches_warmup_and_enforces_request_budget(
    panel: exp.CapacityPanel,
) -> None:
    """SCENARIO-CL-7198-WARMUP: Arms share initialized evidence in every cell."""

    assert len(panel.rows) == len(exp.STREAM_SEEDS) * 12 * len(exp.ARMS)
    assert len(panel.information_budget_rows) == len(panel.rows)
    for seed in exp.STREAM_SEEDS:
        for capacity in exp.CAPACITIES:
            for delay in exp.DELAY_SCHEDULES:
                receipts = [
                    row
                    for row in panel.warmup_state_rows
                    if row["seed"] == seed
                    and row["capacity"] == capacity
                    and row["delay_schedule"] == delay
                ]
                assert len(receipts) == len(exp.ARMS)
                assert len({row["state_hash"] for row in receipts}) == 1
                assert len({tuple(row["released_event_ids"]) for row in receipts}) == 1
                assert all(row["pending_count"] == 0 for row in receipts)
    assert all(row["requests"] <= row["eligible_block_count"] for row in panel.rows)
    assert all(row["max_pending"] <= row["capacity"] for row in panel.rows)
    assert all(row["max_memory_bytes"] <= exp.MEMORY_BYTE_BUDGET for row in panel.rows)
    assert all(row["pending_eviction_count"] == 0 for row in panel.rows)


def test_panel_conformance_names_each_isolated_budget_failure(
    panel: exp.CapacityPanel,
) -> None:
    """SCENARIO-CL-7198-MUTATIONS: Panel bounds fail independently."""

    assert "row_panel" in exp._panel_conformance_errors(replace(panel, rows=panel.rows[:-1]))

    rows = deepcopy(panel.rows)
    rows[0]["max_pending"] = rows[0]["capacity"] + 1
    assert "pending_capacity" in exp._panel_conformance_errors(replace(panel, rows=rows))

    assert "information_budget_count" in exp._panel_conformance_errors(
        replace(panel, information_budget_rows=panel.information_budget_rows[:-1])
    )

    warmup = deepcopy(panel.warmup_state_rows)
    warmup[0]["state_hash"] = "sha256:" + "0" * 64
    assert "warmup_information_mismatch" in exp._panel_conformance_errors(
        replace(panel, warmup_state_rows=warmup)
    )

    pending = deepcopy(panel.pending_queue_rows)
    pending[0]["predictions_committed"] = 3
    assert "queue_chronology" in exp._panel_conformance_errors(
        replace(panel, pending_queue_rows=pending)
    )


def test_release_frees_capacity_only_at_next_boundary(panel: exp.CapacityPanel) -> None:
    """SCENARIO-CL-7198-CAPACITY: Selection sees pre-release occupancy."""

    rows = [
        row
        for row in panel.pending_queue_rows
        if row["seed"] == exp.STREAM_SEEDS[0]
        and row["arm"] == "random_admission"
        and row["capacity"] == 1
        and row["delay_schedule"] == "constant_4"
        and row["chronology_index"] >= exp.WARMUP_COUNT
    ]
    admitted = next(row for row in rows if row["request_status"] == "admitted")
    following = next(row for row in rows if row["block_index"] == admitted["block_index"] + 1)
    assert following["occupancy_before_selection"] == 1
    assert following["request_status"] == "dropped_capacity_full"
    assert following["released_after_selection_count"] >= 1
    after_release = next(row for row in rows if row["block_index"] == admitted["block_index"] + 2)
    assert after_release["occupancy_before_selection"] == 0


def test_random_and_disagreement_use_same_tie_rule_without_future_delay(
    views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7198-CAPACITY: Admission differs only by public scoring."""

    block = views.public_events[128:132]
    ranks = exp.seeded_tie_ranks(exp.STREAM_SEEDS[0], 32, block)
    random_pick = exp.select_request(block, "random_admission", ranks)
    disagreement_pick = exp.select_request(block, "disagreement_admission", ranks)
    assert random_pick in block
    assert disagreement_pick in block
    assert set(ranks) == {row["event_id"] for row in block}
    assert all(exp._nested_keys(row).isdisjoint(exp.FORBIDDEN_PUBLIC_FIELDS) for row in block)
    scores = {row["event_id"]: exp.disagreement_fraction(row) for row in block}
    assert scores[disagreement_pick["event_id"]] == max(scores.values())


def test_headroom_keeps_old_ceiling_and_new_oracle_distinct(
    panel: exp.CapacityPanel,
) -> None:
    """SCENARIO-CL-7198-HEADROOM: The old zero-error result stays no-headroom."""

    old = next(row for row in panel.headroom_rows if row["slice"] == "exp7184_static_future")
    assert old["static_error_rate"] == 0.0
    assert old["oracle_error_rate"] == 0.0
    assert old["headroom"] == 0.0
    assert old["headroom_class"] == "no_headroom"
    new_rows = [row for row in panel.headroom_rows if row["slice"] == "prospective_shifted"]
    assert new_rows
    assert all(row["oracle_is_unattainable"] for row in new_rows)
    assert all(row["oracle_error_rate"] == 0.0 for row in new_rows)
    assert any(row["headroom"] > 0.0 for row in new_rows)


def test_preconditions_parse_real_files_and_reject_quarantine(tmp_path: Path) -> None:
    """SCENARIO-CL-7198-PRECONDITIONS: File gates retain the prior null and quarantine."""

    checks, upstream = exp.collect_preconditions(
        REPO_ROOT,
        exp.DEFAULT_UPSTREAM_ARTIFACT_PATH,
        exp.ExperimentPaths.under(tmp_path),
    )
    assert all(row["passed"] for row in checks)
    null_gate = next(row for row in checks if row["check"] == "known_upstream_null_preserved")
    assert null_gate["expected_value"] == null_gate["observed_value"] == 0
    quarantine = next(row for row in checks if row["check"] == "upstream_not_quarantined")
    assert quarantine["expected_value"] is False
    assert quarantine["observed_value"] is False
    assert upstream["memory_value_score"] == 0

    fixture = deepcopy(upstream)
    fixture["artifact_quarantined"] = True
    fixture_path = tmp_path / "quarantined.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    quarantine_checks, _ = exp.collect_preconditions(
        REPO_ROOT,
        fixture_path,
        exp.ExperimentPaths.under(tmp_path / "blocked"),
    )
    failed = next(row for row in quarantine_checks if row["check"] == "upstream_not_quarantined")
    assert failed["passed"] is False
    assert failed["observed_value"] is True


def test_real_gate_evaluator_blocks_failed_and_promoted_upstream_values(tmp_path: Path) -> None:
    """SCENARIO-CL-7198-PRECONDITIONS: Parsed fixtures use exact completion and null gates."""

    base = json.loads((REPO_ROOT / exp.DEFAULT_UPSTREAM_ARTIFACT_PATH).read_text(encoding="utf-8"))
    failed = deepcopy(base)
    failed["memory_run_complete_score"] = 0
    failed_path = tmp_path / "failed.json"
    failed_path.write_text(json.dumps(failed), encoding="utf-8")
    checks, _ = exp.collect_preconditions(
        REPO_ROOT,
        failed_path,
        exp.ExperimentPaths.under(tmp_path / "failed-output"),
    )
    gate = next(row for row in checks if row["check"] == "upstream_completion_gate")
    assert gate["passed"] is False

    promoted = deepcopy(base)
    promoted["memory_value_score"] = 1
    promoted_path = tmp_path / "promoted.json"
    promoted_path.write_text(json.dumps(promoted), encoding="utf-8")
    checks, _ = exp.collect_preconditions(
        REPO_ROOT,
        promoted_path,
        exp.ExperimentPaths.under(tmp_path / "promoted-output"),
    )
    gate = next(row for row in checks if row["check"] == "known_upstream_null_preserved")
    assert gate["passed"] is False


def test_blocked_artifact_is_terminal_diagnostic_and_row_free(tmp_path: Path) -> None:
    """SCENARIO-CL-7198-PRECONDITIONS: External absence creates a blocked artifact."""

    paths = exp.ExperimentPaths.under(tmp_path)
    checks, upstream = exp.collect_preconditions(
        tmp_path / "missing-repository",
        Path("results/missing.json"),
        paths,
    )
    blocked = exp.build_blocked_artifact(
        checks,
        upstream,
        repo_root=tmp_path / "missing-repository",
        upstream_artifact_path=Path("results/missing.json"),
        paths=paths,
        run_date=exp.RUN_DATE,
        duration_s=0.1,
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["stream_capacity_ready_score"] == 0
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["failed_check"]
    assert blocked["gate_check_summary"]["upstream"]
    assert blocked["gate_check_summary"]["field"]
    assert exp.validate_artifact(blocked) == []
    assert exp.validate_artifact({})[0].startswith("missing_fields:")

    from_runner = exp.build_and_seal(
        tmp_path / "missing-repository",
        paths,
        upstream_artifact_path=Path("results/missing.json"),
        run_date=exp.RUN_DATE,
        duration_s=0.2,
    )
    assert from_runner["verdict_class"] == "blocked"


def test_mutations_are_detected_and_own_readiness_rows(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7198-MUTATIONS: Leakage, overlap, quota, eviction, and bytes fail."""

    assert artifact["stream_capacity_ready_score"] == 1
    assert {row["mutation"] for row in artifact["mutation_audit_rows"]} == {
        "changed_seed",
        "public_hidden_field",
        "overlapping_window",
        "extra_request",
        "pending_eviction",
        "memory_overflow",
    }
    assert all(row["detected"] for row in artifact["mutation_audit_rows"])

    mutations = {
        "changed_seed": ("stream_manifest", "stream_manifest_mismatch"),
        "public_hidden_field": ("public_view_hash", "public_view_hash_mismatch"),
        "overlapping_window": ("stream_manifest", "stream_manifest_mismatch"),
        "extra_request": ("rows", "row_panel_mismatch"),
        "pending_eviction": ("rows", "row_panel_mismatch"),
        "memory_overflow": ("rows", "row_panel_mismatch"),
    }
    for name, (field, expected_error) in mutations.items():
        changed = deepcopy(artifact)
        if name == "changed_seed":
            changed[field]["seeds"][0] += 1
        elif name == "public_hidden_field":
            changed[field] = "sha256:" + "0" * 64
        elif name == "overlapping_window":
            changed[field]["windows"][1]["start"] = 127
        elif name == "extra_request":
            changed[field][0]["requests"] = changed[field][0]["eligible_block_count"] + 1
        elif name == "pending_eviction":
            changed[field][0]["pending_eviction_count"] = 1
        else:
            changed[field][0]["max_memory_bytes"] = exp.MEMORY_BYTE_BUDGET + 1
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected_error in exp.validate_artifact(changed)


def test_complete_artifact_has_required_receipts_and_cold_validation(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7198-TERMINAL: Complete readiness makes no learning claim."""

    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260911"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["verdict_class"] == "circular_positive"
    assert "no learning benefit" in str(artifact["honest_verdict"])
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["sample_size_budget"]["planned_events"] == 10_240
    assert artifact["sample_size_budget"]["completed_events"] == 10_240
    assert artifact["sample_size_budget"]["exclusions"] == []
    assert (
        Path(str(artifact["checkpoint_path"])).name
        != Path(str(artifact["public_stream_path"])).name
    )
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT, check_files=True) == []


def test_immutable_views_reject_changed_bytes(tmp_path: Path) -> None:
    """SCENARIO-CL-7198-STREAM: A sealed view cannot change on a rerun."""

    path = tmp_path / "sealed.jsonl"
    first = exp.write_immutable_jsonl(path, [{"event_id": "one"}])
    assert exp.write_immutable_jsonl(path, [{"event_id": "one"}]) == first
    with pytest.raises(exp.ImmutableSealError, match="immutable_seal_mismatch"):
        exp.write_immutable_jsonl(path, [{"event_id": "two"}])


def test_command_writes_private_terminal_artifact(tmp_path: Path) -> None:
    """REQ-CL-7198: The executable path writes views then one terminal result."""

    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)]) == 0
    paths = exp.ExperimentPaths.under(tmp_path)
    result = json.loads(paths.artifact.read_text(encoding="utf-8"))
    assert result["stream_capacity_ready_score"] == 1
    assert paths.public_stream.is_file()
    assert paths.authority_sidecar.is_file()
    assert paths.checkpoint.parent.name == "checkpoints"
    assert exp.validate_artifact(result, repo_root=REPO_ROOT, check_files=True) == []


def test_builder_refuses_failed_internal_cold_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7198-TERMINAL: Builder validation fails before publication."""

    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced_error"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced_error"):
        exp.build_and_seal(
            REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path),
            run_date=exp.RUN_DATE,
            duration_s=1.25,
        )


def test_command_refuses_an_invalid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7198-TERMINAL: Cold validation precedes the final atomic write."""

    monkeypatch.setattr(
        exp, "build_and_seal", lambda *_args, **_kwargs: {"verdict_class": "blocked"}
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced_error"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced_error"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)])
    assert not exp.ExperimentPaths.under(tmp_path).artifact.exists()
