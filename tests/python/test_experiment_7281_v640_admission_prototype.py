"""Verify fresh paired admission before constraint-memory reuse.

Spec refs: REQ-CL-7281 and SCENARIO-CL-7281-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_7281_v640_admission_prototype as exp


def _masks(parameter: int) -> dict[str, int]:
    """Build one exact finite state without evaluator-only fields."""

    return {family: 1 << parameter for family in exp.FAMILIES}


def _case(
    event_id: str,
    value: int,
    label: str,
    release_index: int,
    *,
    comparison_id: str = "incumbent",
) -> dict[str, object]:
    """Build one released case that scores both frozen states."""

    return {
        "event_id": event_id,
        "family_id": "lower_bound",
        "numeric_value": value,
        "observed_label": label,
        "release_index": release_index,
        "comparison_id": comparison_id,
    }


def _useful_cases(count: int, release_index: int = 30) -> list[dict[str, object]]:
    """Find public cases where parameter eight beats parameter zero."""

    rows = []
    for value in exp.PARAMETER_DOMAIN:
        label = exp.exact_label("lower_bound", value, 8)
        incumbent = exp.prototype.predict_masks(
            _masks(0), {"family_id": "lower_bound", "numeric_value": value}
        )
        if incumbent != label:
            rows.append(_case(f"a-{len(rows)}", value, label, release_index))
        if len(rows) == count:
            break
    assert len(rows) == count
    return rows


@pytest.fixture(scope="module")
def one_stream_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[exp.ExperimentPaths, dict[str, object]]:
    """Build one complete artifact once for terminal and mutation checks."""

    paths = exp.ExperimentPaths.under(tmp_path_factory.mktemp("exp7281-artifact"))
    artifact = exp.build_and_seal(
        exp.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        historical_paths=None,
        progress=True,
    )
    return paths, artifact


def test_req_cl_7281_fixes_contract_and_no_model_work() -> None:
    """REQ-CL-7281 fixes the streams, rules, quotas, state cap, and substrate."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-CL-7281" in spec
    assert len(set(exp.SCENARIO_PATTERN.findall(spec))) == 9
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert exp.INVOCATION_COUNTS == {
        "attempted_model_loads": 0,
        "completed_model_loads": 0,
        "attempted_generation_calls": 0,
        "completed_generation_calls": 0,
        "usable_answers": 0,
    }
    assert exp.ARMS == (
        "full_reference",
        "reset",
        "frozen_warmup",
        "unconditional_recognition",
        "range_gated",
        "paired_gated",
        "label_shuffled_paired",
    )
    assert (exp.DEVELOPMENT_STREAM_COUNT, exp.STREAM_COUNT) == (4, 24)
    assert (exp.EVENTS_PER_STREAM, exp.WARMUP_COUNT) == (1_024, 128)
    assert (exp.NOMINATION_LABEL_BUDGET, exp.ADMISSION_LABEL_BUDGET) == (64, 64)
    assert exp.MAX_OPPORTUNITIES * exp.FRESH_LABELS_PER_OPPORTUNITY == 64
    assert exp.ARCHIVE_CAP == 4
    assert exp.MEMORY_CAP_BYTES == 69_632
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_7281_finite_law_enumerates_n8_n16() -> None:
    """SCENARIO-CL-7281-FINITE-LAW checks exact IID coverage and limits."""

    rows = exp.enumerate_finite_laws()

    assert {row["n"] for row in rows} == {8, 16}
    assert {row["case"] for row in rows} >= {
        "zero_disagreement",
        "all_harmful",
        "all_useful",
        "mixed_disagreement",
    }
    assert all(row["enumerated_probability"] == pytest.approx(1.0) for row in rows)
    assert all(row["noncoverage_probability"] <= row["allocated_alpha"] + 1e-12 for row in rows)
    assert all(row["iid_binary_scope_only"] is True for row in rows)
    assert all(row["dependent_drift_theorem_claimed"] is False for row in rows)
    zero8 = next(row for row in rows if row["case"] == "zero_disagreement" and row["n"] == 8)
    assert zero8["paired_threshold_zero_feasible"] is False
    assert zero8["range_threshold_zero_feasible"] is False


def test_scenario_cl_7281_paired_and_range_decisions_are_distinct() -> None:
    """SCENARIO-CL-7281-CONTROLS keeps rule-specific decisions visible."""

    alpha = exp.opportunity_alpha(1)
    useful16 = [1] * 16
    harmful16 = [-1] * 16
    zero8 = [0] * 8

    paired = exp.score_differences(useful16, alpha, 1, threshold=0.2, rule="paired")
    ranged = exp.score_differences(useful16, alpha, 1, threshold=0.2, rule="range")
    harmful = exp.score_differences(harmful16, alpha, 1, threshold=0.0, rule="paired")
    zero = exp.score_differences(zero8, alpha, 1, threshold=0.0, rule="paired")

    assert paired["decision"] == "accept"
    assert ranged["decision"] == "defer"
    assert harmful["decision"] == "reject"
    assert zero["decision"] == "defer"
    with pytest.raises(ValueError, match="invalid_rule"):
        exp.score_differences(zero8, alpha, 1, threshold=0.0, rule="unknown")
    with pytest.raises(ValueError, match="invalid_comparison_count"):
        exp.score_differences(zero8, alpha, 0, threshold=0.0, rule="paired")


def test_scenario_cl_7281_admission_requires_fresh_disjoint_labels(tmp_path: Path) -> None:
    """SCENARIO-CL-7281-ADMISSION freezes states and rejects label reuse."""

    controller = exp.AdmissionController.from_masks(_masks(0), rule="paired")
    parent_hash = controller.state_hash()
    nomination = controller.nominate(
        _masks(8),
        nomination_event_ids=["n0", "n1"],
        nomination_index=20,
        opportunity_index=1,
        thresholds={"incumbent": -0.5},
    )
    assert nomination["incumbent_state_hash"] == exp.mask_hash(_masks(0))
    assert nomination["candidate_state_hash"] == exp.mask_hash(_masks(8))
    assert controller.incumbent_hash() == exp.mask_hash(_masks(0))
    state_after_nomination = controller.state_bytes()

    reused = _useful_cases(8)
    reused[0]["event_id"] = "n0"
    with pytest.raises(exp.AdmissionRejected, match="reused_nomination_label"):
        controller.admit(reused, current_index=30, expected_parent_hash=controller.state_hash())
    assert controller.state_bytes() == state_after_nomination

    unreleased = _useful_cases(8, release_index=31)
    with pytest.raises(exp.AdmissionRejected, match="unreleased_label"):
        controller.admit(
            unreleased,
            current_index=30,
            expected_parent_hash=controller.state_hash(),
        )
    assert controller.state_bytes() == state_after_nomination

    cases = _useful_cases(8)
    receipt = controller.admit(
        cases,
        current_index=30,
        expected_parent_hash=controller.state_hash(),
        state_path=tmp_path / "state.json",
    )
    assert receipt["decision"] == "accept"
    assert controller.incumbent_hash() == exp.mask_hash(_masks(8))
    assert len(controller.archives()) == 1
    restored = exp.AdmissionController.load(tmp_path / "state.json")
    assert restored.state_bytes() == controller.state_bytes()
    rollback = restored.rollback(receipt, state_path=tmp_path / "state.json")
    assert rollback["byte_identical"] is True
    assert restored.state_hash() == parent_hash


def test_scenario_cl_7281_multiple_comparisons_and_invalid_states() -> None:
    """SCENARIO-CL-7281-ADMISSION applies every fixed comparison together."""

    controller = exp.AdmissionController.from_masks(_masks(0), rule="paired")
    controller.nominate(
        _masks(8),
        nomination_event_ids=["n"],
        nomination_index=20,
        opportunity_index=1,
        thresholds={"incumbent": -0.5, "protected": -0.5},
    )
    cases = _useful_cases(8)
    cases.extend(
        _case(
            f"p-{index}",
            int(row["numeric_value"]),
            str(row["observed_label"]),
            30,
            comparison_id="protected",
        )
        for index, row in enumerate(_useful_cases(8))
    )
    receipt = controller.admit(
        cases,
        current_index=30,
        expected_parent_hash=controller.state_hash(),
    )
    assert receipt["comparison_count"] == 2
    assert {row["comparison_id"] for row in receipt["comparison_rows"]} == {
        "incumbent",
        "protected",
    }

    with pytest.raises(ValueError, match="invalid_admission_rule"):
        exp.AdmissionController.from_masks(_masks(0), rule="bad")
    broken = controller.state_dict()
    broken["archives"] = [{"state_hash": "bad", "masks": _masks(0)}]
    with pytest.raises(ValueError, match="archive_identity"):
        exp.AdmissionController.from_state(broken)


def test_scenario_cl_7281_old_rows_measure_overlap_without_tuning(tmp_path: Path) -> None:
    """SCENARIO-CL-7281-OLD-EVIDENCE audits overlap and later error only."""

    lifecycle = tmp_path / "lifecycle.jsonl"
    event_rows = tmp_path / "events.jsonl"
    lifecycle_rows = [
        {
            "kind": "nomination",
            "stream_id": "s",
            "arm": "paired_gated",
            "event_id": "e1",
            "chronology_index": 1,
        },
        {
            "kind": "validation",
            "stream_id": "s",
            "arm": "paired_gated",
            "event_id": "e1",
            "chronology_index": 1,
        },
        {
            "kind": "reactivation",
            "stream_id": "s",
            "arm": "paired_gated",
            "event_id": "e2",
            "chronology_index": 2,
        },
    ]
    rows = []
    for index in range(4):
        rows.extend(
            [
                {
                    "stream_id": "s",
                    "arm": "frozen",
                    "chronology_index": index,
                    "full_denominator_error": 0,
                },
                {
                    "stream_id": "s",
                    "arm": "paired_gated",
                    "chronology_index": index,
                    "full_denominator_error": int(index == 3),
                },
            ]
        )
    lifecycle.write_bytes(exp.prototype.jsonl_bytes(lifecycle_rows))
    event_rows.write_bytes(exp.prototype.jsonl_bytes(rows))

    overlap = exp.reduce_old_lifecycle(lifecycle, event_rows)

    assert overlap[0]["nomination_validation_overlap_count"] == 1
    assert overlap[0]["nomination_validation_overlap_rate"] == 1.0
    assert overlap[0]["harmful_reactivation_count"] == 1
    assert overlap[0]["old_outcomes_used_for_rule_tuning"] is False


def test_scenario_cl_7281_streams_are_new_complete_and_sealed(tmp_path: Path) -> None:
    """SCENARIO-CL-7281-STREAMS seals separate public and evaluator views."""

    development = exp.build_stream_views("development")
    prospective = exp.build_stream_views("prospective")

    assert exp.stream_conformance_errors(development, "development") == []
    assert exp.stream_conformance_errors(prospective, "prospective") == []
    assert prospective.manifest["strata"] == {
        "separated_recurrence": 12,
        "overlapping_recurrence": 12,
    }
    assert len({row["event_id"] for row in prospective.public}) == 24 * 1_024
    assert not exp.prototype.public_leakage_errors(prospective.public)
    paths = exp.ExperimentPaths.under(tmp_path)
    manifest = exp.seal_streams(paths, development, prospective)
    assert manifest["authority_separated"] is True
    assert Path(manifest["manifest_receipt"]["path"]).is_file()

    leaked = deepcopy(prospective)
    leaked.public[0]["regime_id"] = "private"
    assert exp.stream_conformance_errors(leaked, "prospective") == ["public_authority_leakage"]
    with pytest.raises(ValueError, match="invalid_stream_kind"):
        exp.build_stream_views("old")


def test_scenario_cl_7281_panel_preserves_quotas_controls_and_rows() -> None:
    """SCENARIO-CL-7281-QUOTAS executes all arms on equal paid evidence."""

    views = exp.build_stream_views("prospective")
    panel = exp.run_admission_panel(views, stream_ids=("prospective-01",), progress=True)

    assert len(panel.rows) == len(exp.ARMS)
    assert {row["arm"] for row in panel.rows} == set(exp.ARMS)
    assert exp.opportunity_row_errors(panel.opportunity_rows) == []
    assert all(row["event_count"] == 1_024 for row in panel.rows)
    assert all(row["nomination_label_count"] == 64 for row in panel.rows)
    assert all(row["admission_label_count"] == 64 for row in panel.rows)
    assert all(row["total_paid_label_count"] == 128 for row in panel.rows)
    assert all(row["maximum_memory_bytes"] <= exp.MEMORY_CAP_BYTES for row in panel.rows)
    assert len({row["admission_case_ids_sha256"] for row in panel.opportunity_rows}) == 8
    assert panel.censored_stream_count == 0


def test_scenario_cl_7281_development_and_e2e_controls(tmp_path: Path) -> None:
    """SCENARIO-CL-7281-E2E proves intervention, restart, rejection, and rollback."""

    development = exp.run_development_controls(exp.build_stream_views("development"))
    mutations = exp.run_mutation_controls(tmp_path / "mutations")
    e2e = exp.run_e2e_controls(tmp_path / "e2e")

    assert development["control_decision_change_count"] > 0
    assert development["parameters_frozen_before_prospective"] is True
    assert {row["mutation"] for row in mutations} == {"reused_label", "unreleased_label"}
    assert all(row["passed"] is True for row in mutations)
    assert {row["stage"] for row in e2e} == {
        "public_prediction",
        "immutable_nomination",
        "disjoint_delayed_feedback",
        "admission",
        "later_prediction",
        "cold_restart",
        "rejection",
        "rollback",
    }
    assert all(row["passed"] is True for row in e2e)


def test_scenario_cl_7281_blocked_external_input_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-CL-7281-PRECONDITIONS makes external absence terminal and row-free."""

    paths = exp.ExperimentPaths.under(tmp_path / "out")
    checks, hashes = exp.collect_preconditions(
        exp.REPO_ROOT,
        paths,
        upstream_overrides={"exp7268": tmp_path / "missing.json"},
    )
    artifact = exp.build_blocked_artifact(
        checks,
        hashes,
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.01,
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["admission_fixture_ready_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["observed_value"] is None
    assert exp.validate_artifact(artifact, expected_stream_ids=("prospective-01",)) == []


def test_scenario_cl_7281_terminal_build_is_valid_and_atomic(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7281-TERMINAL seals measured rows only after validation."""

    paths, artifact = one_stream_artifact

    assert artifact["status"] == "complete"
    assert artifact["admission_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_circular_positive")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["bound_infeasibility_reported"] is True
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
    assert json.loads(paths.artifact.read_text(encoding="utf-8"))["experiment_id"] == 7281


def test_defensive_validation_receipts_and_thin_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7281 rejects malformed terminal evidence and keeps the wrapper thin."""

    receipt = {
        "command": "focused-test",
        "exit_code": 0,
        "classification": "passed",
        "duration_s": 0.1,
        "log_sha256": "sha256:" + "1" * 64,
    }
    blocked = exp.build_blocked_artifact(
        [exp.gate_check("x", "upstream", "field", 1, 0)],
        {},
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.01,
    )
    changed = exp.attach_validation_receipts(blocked, [receipt])
    assert changed["validation_receipts"] == [receipt]
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp.attach_validation_receipts(blocked, [{"command": "bad"}])
    invalid = deepcopy(changed)
    invalid["model_invoked"] = True
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    assert "model_contract" in exp.validate_artifact(invalid)

    called: list[object] = []

    def fake_main(argv: object = None) -> int:
        called.append(argv)
        return 0

    monkeypatch.setattr(exp, "main", fake_main)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(exp.REPO_ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0
    assert called == [None]


def test_cli_date_and_private_e2e_paths(tmp_path: Path) -> None:
    """REQ-CL-7281 keeps CLI failures explicit and test writes private."""

    with pytest.raises(SystemExit, match="run_date_must_be_20260913"):
        exp.main(["--date", "20260912", "--output-root", str(tmp_path)])
    assert exp.main(["--date", "20260913", "--output-root", str(tmp_path), "--e2e-worker"]) == 0
    assert (tmp_path / "checkpoints" / "experiment_7281_v640_e2e.json").is_file()


def test_controller_rejects_every_malformed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7281-ADMISSION rejects malformed states and evidence."""

    with pytest.raises(ValueError, match="invalid_mask_families"):
        exp.AdmissionController.from_masks({})
    bad_masks = _masks(0)
    bad_masks[exp.FAMILIES[0]] = 0
    with pytest.raises(ValueError, match="invalid_survivor_mask"):
        exp.AdmissionController.from_masks(bad_masks)
    with pytest.raises(ValueError, match="invalid_opportunity_index"):
        exp.opportunity_alpha(0)
    with pytest.raises(ValueError, match="invalid_binomial_inputs"):
        exp._clopper_pearson(0, 0, 0.1)
    with pytest.raises(ValueError, match="invalid_paired_differences"):
        exp.score_differences([], 0.1, 1, threshold=0.0, rule="paired")

    controller = exp.AdmissionController.from_masks(_masks(0))
    with pytest.raises(exp.AdmissionRejected, match="invalid_nomination_ids"):
        controller.nominate(
            _masks(8),
            nomination_event_ids=[],
            nomination_index=20,
            opportunity_index=1,
            thresholds={"incumbent": -0.5},
        )
    with pytest.raises(exp.AdmissionRejected, match="missing_comparisons"):
        controller.nominate(
            _masks(8),
            nomination_event_ids=["n"],
            nomination_index=20,
            opportunity_index=1,
            thresholds={},
        )
    with pytest.raises(ValueError, match="invalid_admission_state"):
        exp.AdmissionController.from_state({})
    state = controller.state_dict()
    state["rule"] = "bad"
    with pytest.raises(ValueError, match="invalid_admission_rule"):
        exp.AdmissionController.from_state(state)
    state = controller.state_dict()
    state["archives"] = [
        {"state_hash": exp.mask_hash(_masks(index)), "masks": _masks(index)} for index in range(5)
    ]
    with pytest.raises(ValueError, match="archive_capacity"):
        exp.AdmissionController.from_state(state)

    pending = exp.AdmissionController.from_masks(_masks(0))
    pending.nominate(
        _masks(8),
        nomination_event_ids=["n"],
        nomination_index=20,
        opportunity_index=1,
        thresholds={"incumbent": -0.5},
    )
    broken = pending.state_dict()
    broken["pending"]["candidate_state_hash"] = "bad"
    with pytest.raises(ValueError, match="pending_identity"):
        exp.AdmissionController.from_state(broken)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_admission_state"):
        exp.AdmissionController.load(invalid_json)
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_admission_state"):
        exp.AdmissionController.load(invalid_json)
    saved = controller.save(tmp_path / "saved.json")
    assert saved["sha256"] == exp._sha256_path(tmp_path / "saved.json")

    monkeypatch.setattr(
        exp.AdmissionController,
        "memory_usage",
        lambda self: {"within_cap": False},
    )
    with pytest.raises(ValueError, match="admission_memory_cap"):
        exp.AdmissionController.from_state(controller.state_dict())
    with pytest.raises(exp.AdmissionRejected, match="admission_memory_cap"):
        controller.nominate(
            _masks(8),
            nomination_event_ids=["n"],
            nomination_index=20,
            opportunity_index=1,
            thresholds={"incumbent": -0.5},
        )


def test_admission_rejects_duplicate_stale_and_incomplete_cases() -> None:
    """SCENARIO-CL-7281-ADMISSION rejects all freshness contract violations."""

    def pending_controller() -> exp.AdmissionController:
        controller = exp.AdmissionController.from_masks(_masks(0))
        controller.nominate(
            _masks(8),
            nomination_event_ids=["n"],
            nomination_index=20,
            opportunity_index=1,
            thresholds={"incumbent": -0.5},
        )
        return controller

    no_pending = exp.AdmissionController.from_masks(_masks(0))
    with pytest.raises(exp.AdmissionRejected, match="no_pending_decision"):
        no_pending._validated_cases([], 30)

    pending = pending_controller()
    with pytest.raises(exp.AdmissionRejected, match="pending_decision_exists"):
        pending.nominate(
            _masks(8),
            nomination_event_ids=["again"],
            nomination_index=21,
            opportunity_index=1,
            thresholds={"incumbent": -0.5},
        )
    for expected, mutation in (
        ("duplicate_admission_label", "duplicate"),
        ("label_not_subsequent", "old"),
        ("invalid_admission_case", "invalid"),
        ("comparison_evidence_mismatch", "comparison"),
    ):
        controller = pending_controller()
        cases = _useful_cases(8)
        if mutation == "duplicate":
            cases[1]["event_id"] = cases[0]["event_id"]
        elif mutation == "old":
            cases[0]["release_index"] = 20
        elif mutation == "invalid":
            cases[0]["observed_label"] = "unknown"
        else:
            cases[0]["comparison_id"] = "other"
        with pytest.raises(exp.AdmissionRejected, match=expected):
            controller.admit(
                cases,
                current_index=30,
                expected_parent_hash=controller.state_hash(),
            )
    stale = pending_controller()
    with pytest.raises(exp.AdmissionRejected, match="stale_parent"):
        stale.admit(_useful_cases(8), current_index=30, expected_parent_hash="bad")

    used = pending_controller()
    receipt = used.admit(_useful_cases(8), current_index=30, expected_parent_hash=used.state_hash())
    used.nominate(
        _masks(0),
        nomination_event_ids=["n2"],
        nomination_index=40,
        opportunity_index=2,
        thresholds={"incumbent": -0.5},
    )
    reused = _useful_cases(8, release_index=50)
    with pytest.raises(exp.AdmissionRejected, match="reused_admission_label"):
        used.admit(reused, current_index=50, expected_parent_hash=used.state_hash())
    with pytest.raises(exp.AdmissionRejected, match="stale_rollback"):
        pending_controller().rollback(receipt)

    invalid_receipt_controller = exp.AdmissionController.from_masks(_masks(0))
    with pytest.raises(exp.AdmissionRejected, match="invalid_rollback_receipt"):
        invalid_receipt_controller.rollback(
            {
                "new_state_hash": invalid_receipt_controller.state_hash(),
                "parent_bytes_b64": "bad",
            }
        )
    with pytest.raises(exp.AdmissionRejected, match="rollback_parent_hash"):
        invalid_receipt_controller.rollback(
            {
                "new_state_hash": invalid_receipt_controller.state_hash(),
                "parent_bytes_b64": exp.transactional.encode_bytes(
                    invalid_receipt_controller.state_bytes()
                ),
                "pre_nomination_parent_hash": "bad",
            }
        )


def test_stream_row_and_historical_defensive_checks(tmp_path: Path) -> None:
    """SCENARIO-CL-7281-STREAMS rejects incomplete, leaked, and malformed rows."""

    views = exp.build_stream_views("development")
    changed = deepcopy(views)
    changed.public.pop()
    assert exp.stream_conformance_errors(changed, "development") == [
        "event_count",
        "event_identity",
        "chronology",
    ]
    changed = deepcopy(views)
    changed.authority[0]["event_id"] = "changed"
    assert exp.stream_conformance_errors(changed, "development") == ["event_identity"]
    changed = deepcopy(views)
    changed.manifest["strata"] = {}
    assert exp.stream_conformance_errors(changed, "development") == ["strata"]
    changed = deepcopy(views)
    changed.public[0]["chronology_index"] = 2
    assert exp.stream_conformance_errors(changed, "development") == ["chronology"]
    with pytest.raises(ValueError, match="incomplete_stream"):
        exp.run_admission_panel(views, stream_ids=("prospective-01",))

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_raw_rows"):
        exp.independent_reduce(empty)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="historical_lifecycle_unavailable"):
        exp.reduce_old_lifecycle(malformed, empty)
    lifecycle = tmp_path / "lifecycle.jsonl"
    lifecycle.write_text(
        json.dumps(
            {
                "kind": "reactivation",
                "stream_id": "s",
                "arm": "a",
                "event_id": "e",
                "chronology_index": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="historical_event_rows_unavailable"):
        exp.reduce_old_lifecycle(lifecycle, malformed)
    lifecycle.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="historical_lifecycle_empty"):
        exp.reduce_old_lifecycle(lifecycle, empty)
    invalid_development = deepcopy(views)
    invalid_development.manifest["strata"] = {}
    with pytest.raises(ValueError, match="invalid_development_streams"):
        exp.run_development_controls(invalid_development)
    with pytest.raises(ValueError, match="insufficient_contrast_cases"):
        exp._contrast_cases(8, 0, count=10_000, start=0, release_index=1)


def test_opportunity_error_categories(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
) -> None:
    """SCENARIO-CL-7281-QUOTAS names each raw opportunity defect."""

    _, artifact = one_stream_artifact
    original = artifact["opportunity_rows"]
    changes = (
        ("label_reuse", "label_overlap_count", 1),
        ("unreleased_label", "all_admission_labels_released", False),
        ("authority_leakage", "private_regime_used", True),
        ("quota", "admission_label_count", 7),
        ("memory_cap", "memory_bytes", exp.MEMORY_CAP_BYTES + 1),
    )
    for expected, field, value in changes:
        rows = deepcopy(original)
        rows[0][field] = value
        assert expected in exp.opportunity_row_errors(rows)
    rows = deepcopy(original)
    rows.pop()
    assert "incomplete_opportunities" in exp.opportunity_row_errors(rows)
    rows = deepcopy(original)
    rows[0]["admission_case_ids_sha256"] = "changed"
    assert "common_case_mismatch" in exp.opportunity_row_errors(rows)


def test_progress_heartbeat_and_build_fail_closed_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7281 keeps long-loop progress and internal failures explicit."""

    views = exp.build_stream_views("prospective")
    clock_calls = 0

    def heartbeat_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 61.0

    monkeypatch.setattr(exp.time, "monotonic", heartbeat_clock)
    exp.run_admission_panel(views, stream_ids=("prospective-01",), progress=True)
    assert "benchmark heartbeat" in capsys.readouterr().out
    monkeypatch.undo()

    paths = exp.ExperimentPaths.under(tmp_path / "blocked")
    failed = exp.gate_check("missing", "upstream", "field", "present", None)
    monkeypatch.setattr(exp, "collect_preconditions", lambda *args: ([failed], {}))
    blocked = exp.build_and_seal(exp.REPO_ROOT, paths, progress=True)
    assert blocked["status"] == "blocked"
    monkeypatch.undo()

    passed = exp.gate_check("present", "upstream", "field", 1, 1)
    monkeypatch.setattr(exp, "collect_preconditions", lambda *args: ([passed], {}))
    monkeypatch.setattr(exp, "reduce_old_lifecycle", lambda *args: [])
    monkeypatch.setattr(exp, "enumerate_finite_laws", lambda: [])
    monkeypatch.setattr(exp, "build_stream_views", lambda kind: object())
    monkeypatch.setattr(exp, "stream_conformance_errors", lambda *args: ["bad_stream"])
    with pytest.raises(ValueError, match="stream_conformance_failed"):
        exp.build_and_seal(exp.REPO_ROOT, exp.ExperimentPaths.under(tmp_path / "stream"))
    monkeypatch.undo()

    panel = exp.AdmissionPanel([], [], [], 0, 0, 0)

    def prepare_short_build() -> None:
        monkeypatch.setattr(exp, "collect_preconditions", lambda *args: ([passed], {}))
        monkeypatch.setattr(exp, "reduce_old_lifecycle", lambda *args: [])
        monkeypatch.setattr(exp, "enumerate_finite_laws", lambda: [])
        monkeypatch.setattr(exp, "build_stream_views", lambda kind: object())
        monkeypatch.setattr(exp, "stream_conformance_errors", lambda *args: [])
        monkeypatch.setattr(exp, "seal_streams", lambda *args: {})
        monkeypatch.setattr(
            exp,
            "run_development_controls",
            lambda *args: {"control_decision_change_count": 1, "rows": []},
        )
        monkeypatch.setattr(exp, "run_admission_panel", lambda *args, **kwargs: panel)

    prepare_short_build()
    monkeypatch.setattr(exp, "opportunity_row_errors", lambda rows: ["bad_opportunity"])
    with pytest.raises(ValueError, match="opportunity_row_errors"):
        exp.build_and_seal(exp.REPO_ROOT, exp.ExperimentPaths.under(tmp_path / "opportunity"))
    monkeypatch.undo()

    prepare_short_build()
    mismatch_panel = exp.AdmissionPanel([{"unit": 1}], [], [], 0, 0, 0)
    monkeypatch.setattr(exp, "run_admission_panel", lambda *args, **kwargs: mismatch_panel)
    monkeypatch.setattr(exp, "opportunity_row_errors", lambda rows: [])
    monkeypatch.setattr(exp, "run_mutation_controls", lambda *args: [])
    monkeypatch.setattr(exp, "run_e2e_controls", lambda *args: [])
    monkeypatch.setattr(
        exp,
        "_atomic_write",
        lambda path, payload: {
            "path": str(path),
            "sha256": exp.transactional.sha256_bytes(payload),
        },
    )
    monkeypatch.setattr(exp, "independent_reduce", lambda path: [])
    with pytest.raises(ValueError, match="independent_reducer_mismatch"):
        exp.build_and_seal(exp.REPO_ROOT, exp.ExperimentPaths.under(tmp_path / "reducer"))


def test_build_rejects_its_own_invalid_terminal_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7281 cold-validates the measured candidate before returning it."""

    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp.build_and_seal(
            exp.REPO_ROOT,
            exp.ExperimentPaths.under(tmp_path),
            stream_ids=("prospective-01",),
        )


def test_validator_writer_commands_and_cli_orchestration(
    one_stream_artifact: tuple[exp.ExperimentPaths, dict[str, object]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7281 covers cold validation, commands, and terminal orchestration."""

    _, artifact = one_stream_artifact
    assert exp.ExperimentPaths.defaults().artifact == exp.REPO_ROOT / exp.DEFAULT_ARTIFACT
    commands = exp._validation_commands(tmp_path / "candidate.json")
    assert any("check_spec_coverage.py" in command for row in commands for command in row)
    command_receipt = exp._command_receipt([sys.executable, "-c", "print('command-receipt-ok')"])
    assert command_receipt["exit_code"] == 0

    invalid = deepcopy(artifact)
    invalid["status"] = "unfinished"
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    assert "status" in exp.validate_artifact(invalid)
    null_artifact = deepcopy(artifact)
    null_artifact["admission_fixture_ready_score"] = 0
    null_artifact["verdict_class"] = "null"
    null_artifact["honest_verdict"] = "complete_null: forced test gate"
    first_gate = next(iter(null_artifact["acceptance_gate_results"].values()))
    first_gate["passed"] = False
    first_gate["pass"] = False
    null_artifact["reproducibility_checksum"] = exp.reproducibility_checksum(null_artifact)
    assert "complete_contract" not in exp.validate_artifact(
        null_artifact, expected_stream_ids=("prospective-01",)
    )
    invalid = deepcopy(artifact)
    invalid["rows"] = []
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp.write_artifact(tmp_path / "bad.json", invalid)

    empty_raw = tmp_path / "empty.jsonl"
    empty_raw.write_text("", encoding="utf-8")
    malformed_files = deepcopy(artifact)
    malformed_files["raw_rows_receipt"] = {
        "path": str(empty_raw),
        "sha256": exp._sha256_path(empty_raw),
        "row_count": len(artifact["rows"]),
    }
    malformed_files["reproducibility_checksum"] = exp.reproducibility_checksum(malformed_files)
    assert "independent_reducer" in exp.validate_artifact(
        malformed_files,
        expected_stream_ids=("prospective-01",),
        check_files=True,
    )

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_bytes(exp.transactional.canonical_json_bytes(artifact))
    real_validate_artifact = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: [])
    assert (
        exp.main(["--date", exp.RUN_DATE, "--validate", "--artifact-path", str(artifact_path)]) == 0
    )
    monkeypatch.setattr(exp, "validate_artifact", real_validate_artifact)
    with pytest.raises(SystemExit, match="artifact_path_required"):
        exp.main(["--date", exp.RUN_DATE, "--validate"])
    artifact_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="artifact_validation_failed"):
        exp.main(["--date", exp.RUN_DATE, "--validate", "--artifact-path", str(artifact_path)])

    class FakeOutput:
        def readline(self) -> str:
            return ""

        def read(self) -> str:
            return "trailing-output\n"

    class FakeProcess:
        stdout = FakeOutput()

        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

        def wait(self) -> int:
            return 0

    class FakeSelector:
        def register(self, *args: object) -> None:
            pass

        def select(self, timeout: float) -> list[object]:
            return []

        def close(self) -> None:
            pass

    with monkeypatch.context() as context:
        context.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
        context.setattr(exp.selectors, "DefaultSelector", FakeSelector)
        heartbeat_receipt = exp._command_receipt(["fake-command"])
    assert heartbeat_receipt["exit_code"] == 0

    blocked = exp.build_blocked_artifact(
        [exp.gate_check("x", "up", "field", 1, 0)],
        {},
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.01,
    )
    written: list[Path] = []
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: blocked)
    monkeypatch.setattr(exp, "write_artifact", lambda path, *args, **kwargs: written.append(path))
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "blocked")]) == 0
    assert written

    complete = deepcopy(artifact)
    monkeypatch.setattr(exp, "build_and_seal", lambda *args, **kwargs: complete)
    monkeypatch.setattr(exp, "_validation_commands", lambda candidate: [["focused-check"]])
    passed_receipt = {
        "command": "focused-check",
        "exit_code": 0,
        "classification": "passed",
        "duration_s": 0.01,
        "log_sha256": exp.transactional.sha256_bytes(b"passed"),
    }
    monkeypatch.setattr(exp, "_command_receipt", lambda command: passed_receipt)
    monkeypatch.setattr(exp, "_atomic_write", lambda path, payload: {"path": str(path)})
    monkeypatch.setattr(exp, "write_artifact", lambda *args, **kwargs: None)
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "complete")]) == 0

    failed_receipt = {
        **passed_receipt,
        "exit_code": 1,
        "classification": "failed",
        "log_sha256": exp.transactional.sha256_bytes(b"failed"),
    }
    monkeypatch.setattr(exp, "_command_receipt", lambda command: failed_receipt)
    with pytest.raises(RuntimeError, match="focused_validation_failed"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "failed")])
