"""Tests for exact certification and calibration-only schedule selection.

Spec refs: REQ-VERIFY-6976 and SCENARIO-VERIFY-6976-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6975_delayed_constraint_candidate_bank as bank_exp
from carnot import experiment_6976_exact_candidate_certification as exp


REPO = Path(__file__).resolve().parents[2]


def _formulation(name: str, *, coefficient: str = "1") -> dict:
    return {
        "schema_version": "carnot.bounded_optimization_formulation.v1",
        "variables": [
            {
                "name": name,
                "kind": "integer",
                "universe": [-1, 0, 1, 2],
                "domain": {"lower": "0", "upper": "1"},
            }
        ],
        "constraints": [{"terms": {name: "1"}, "op": "<=", "rhs": "1"}],
        "objective": {
            "direction": "min",
            "expression": {
                "kind": "linear",
                "terms": {name: coefficient},
                "constant": "0",
            },
        },
    }


def _pair() -> dict:
    return {
        "pair_id": "pair-1",
        "split": "calibration",
        "formulation_family": "bounded_integer_linear",
        "source_formulation": _formulation("x"),
        "target_formulation": _formulation("y"),
    }


def _raw_mapping(*, target: str = "y", scale: str = "1") -> str:
    return json.dumps(
        {
            "schema_version": "carnot.constraint_ir.mapping.v1",
            "variable_map": [{"source": "x", "target": target, "scale": scale, "offset": "0"}],
            "objective_map": {"direction": "same", "scale": scale, "offset": "0"},
        },
        sort_keys=True,
    )


def _attempt(raw_text: str, *, split: str = "calibration") -> dict:
    return {
        "attempt_key": f"model|pair-1|direct|{len(raw_text)}",
        "ordinal": 0,
        "hf_id": "model",
        "pair_id": "pair-1",
        "split": split,
        "formulation_family": "bounded_integer_linear",
        "schedule_id": "direct",
        "call_status": "complete",
        "candidate_raw_text": raw_text,
        "candidate_raw_sha256": bank_exp.sha256_text(raw_text),
        "parser_diagnostic": bank_exp.parse_syntax(raw_text),
        "exception_type": None,
        "exception_message": None,
        "terminal": True,
    }


def _candidate(
    model: str,
    pair_id: str,
    schedule: str,
    success: bool,
    *,
    split: str,
    parsed: bool = True,
) -> dict:
    return {
        "attempt_key": f"{model}|{pair_id}|{schedule}",
        "hf_id": model,
        "pair_id": pair_id,
        "split": split,
        "formulation_family": "family-a" if pair_id != "p2" else "family-b",
        "schedule_id": schedule,
        "parse_success": parsed,
        "exact_semantic_success": success,
        "terminal": True,
    }


@pytest.fixture(scope="module")
def frozen_artifact() -> dict:
    """REQ-VERIFY-6976 runs the frozen 108-row integration path once."""

    artifact = exp.build_from_paths(REPO, date="20260904")
    exp.validate_artifact(artifact)
    return artifact


def test_req_verify_6976_spec_anchors_fields_and_scenarios() -> None:
    """REQ-VERIFY-6976 owns every required field and named failure scenario."""

    text = (REPO / "openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6976") :]
    for name in (
        "PRECONDITIONS",
        "PARSER",
        "SOLVERS",
        "OUTCOMES",
        "CALIBRATION",
        "HELDOUT",
        "HEADROOM",
        "PAIRED",
        "BARE",
    ):
        assert f"SCENARIO-VERIFY-6976-{name}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in {
            "schema",
            "experiment_id",
            "run_date",
        }


def test_scenario_verify_6976_parser_has_exact_frozen_parity() -> None:
    """SCENARIO-VERIFY-6976-PARSER reparses exact bytes without extraction."""

    pair = _pair()
    for raw in (_raw_mapping(), "", "```json\n{}\n```", "[]", "{}"):
        attempt = _attempt(raw)
        parser_row, mapping = exp.parse_raw_candidate(attempt, pair)
        assert parser_row["stored_diagnostic"] == bank_exp.parse_syntax(raw)
        assert parser_row["reparsed_diagnostic"] == parser_row["stored_diagnostic"]
        assert parser_row["parser_parity"] is True
        assert parser_row["raw_sha256_matches"] is True
        assert parser_row["terminal"] is True
        assert (mapping is not None) is parser_row["parse_success"]
    fenced, mapping = exp.parse_raw_candidate(_attempt("```json\n{}\n```"), pair)
    assert fenced["parse_reason"] == "malformed_json"
    assert mapping is None


def test_scenario_verify_6976_parser_exposes_hash_and_diagnostic_drift() -> None:
    """SCENARIO-VERIFY-6976-PARSER keeps stored drift visible and terminal."""

    attempt = _attempt(_raw_mapping())
    attempt["candidate_raw_sha256"] = "sha256:changed"
    attempt["parser_diagnostic"] = {"json_valid": False}
    row, mapping = exp.parse_raw_candidate(attempt, _pair())
    assert row["parser_parity"] is False
    assert row["raw_sha256_matches"] is False
    assert mapping is not None


def test_scenario_verify_6976_solvers_certify_direction_and_solution_space() -> None:
    """SCENARIO-VERIFY-6976-SOLVERS runs both exact authorities on one map."""

    bundle = exp.certify_candidate(_attempt(_raw_mapping()), _pair())
    row = bundle["candidate_row"]
    agreement = bundle["solver_agreement_row"]
    witness = bundle["exact_witness_row"]
    assert row["certified_relation"] == "equivalent"
    assert row["schema_outcome"] == "valid"
    assert row["domain_correspondence_outcome"] == "passed"
    assert row["objective_direction_outcome"] == "passed"
    assert row["objective_order_outcome"] == "passed"
    assert row["optimum_outcome"] == "passed"
    assert row["solution_space_equivalence_outcome"] == "passed"
    assert agreement["all_required_agreement"] is True
    assert agreement["satisfiability_agreement"] is True
    assert agreement["optimum_agreement"] is True
    assert agreement["mapping_direction_agreement"] is True
    assert agreement["solution_space_agreement"] is True
    assert witness["enumeration_witnesses"]
    assert witness["z3_witnesses"]


def test_scenario_verify_6976_outcomes_keep_schema_and_nondecisions_separate() -> None:
    """SCENARIO-VERIFY-6976-OUTCOMES preserves each terminal failure class."""

    schema = exp.certify_candidate(_attempt(_raw_mapping(target="absent")), _pair())
    assert schema["candidate_row"]["schema_outcome"] == "rejected"
    assert schema["candidate_row"]["parse_outcome"] == "parsed"
    assert schema["candidate_row"]["certified_relation"] is None
    assert schema["solver_agreement_row"]["all_required_agreement"] is True

    malformed = exp.certify_candidate(_attempt("{bad"), _pair())
    assert malformed["candidate_row"]["parse_outcome"] == "rejected"
    assert malformed["candidate_row"]["schema_outcome"] == "not_evaluated"

    real = exp.certify_candidate(_attempt(_raw_mapping()), _pair())
    unknown = deepcopy(real["exact_witness_row"]["enumeration_engine_row"])
    unknown.update({"status": "unknown", "label": None, "unknown_reasons": ["injected"]})
    timeout = deepcopy(unknown)
    timeout.update({"status": "timeout", "unknown_reasons": ["timeout"]})

    def unknown_engine(_pair: dict) -> dict:
        return deepcopy(unknown)

    def timeout_engine(_pair: dict) -> dict:
        return deepcopy(timeout)

    nondecision = exp.certify_candidate(
        _attempt(_raw_mapping()),
        _pair(),
        enumeration_certifier=unknown_engine,
        z3_certifier=timeout_engine,
    )["candidate_row"]
    assert nondecision["unknown_outcome"] == {"enumeration": True, "z3": False}
    assert nondecision["timeout_outcome"] == {"enumeration": False, "z3": True}
    assert nondecision["exact_semantic_success"] is None

    def exploding(_pair: dict) -> dict:
        raise RuntimeError("injected solver failure")

    exception = exp.certify_candidate(
        _attempt(_raw_mapping()),
        _pair(),
        enumeration_certifier=exploding,
        z3_certifier=exploding,
    )["candidate_row"]
    assert exception["exception_outcome"] == {"enumeration": True, "z3": True}
    assert exception["terminal"] is True

    shape = exp.certify_candidate(_attempt("{}"), _pair())["candidate_row"]
    assert shape["parse_outcome"] == "parsed"
    assert shape["schema_outcome"] == "rejected"

    def timed_out(_pair: dict) -> dict:
        raise TimeoutError("injected timeout")

    def invalid_status(_pair: dict) -> dict:
        return {"status": "running"}

    timeout_and_invalid = exp.certify_candidate(
        _attempt(_raw_mapping()),
        _pair(),
        enumeration_certifier=timed_out,
        z3_certifier=invalid_status,
    )["candidate_row"]
    assert timeout_and_invalid["timeout_outcome"]["enumeration"] is True
    assert timeout_and_invalid["exception_outcome"]["z3"] is True
    assert exp._combined_outcome([True, None]) == "unresolved"


def test_scenario_verify_6976_calibration_rank_is_label_scoped_and_stable() -> None:
    """SCENARIO-VERIFY-6976-CALIBRATION applies the registered three-part rank."""

    rows = []
    for pair_id in ("p1", "p2"):
        rows.extend(
            [
                _candidate("m", pair_id, "direct", pair_id == "p1", split="calibration"),
                _candidate("m", pair_id, "trigger_switched", True, split="calibration"),
                _candidate(
                    "m",
                    pair_id,
                    "draft_conditioned",
                    True,
                    split="calibration",
                    parsed=False,
                ),
            ]
        )
    metrics, ranking, policy, policy_hash = exp.select_schedule(rows)
    assert policy["schedule_id"] == "trigger_switched"
    assert [row["schedule_id"] for row in ranking] == [
        "trigger_switched",
        "draft_conditioned",
        "direct",
    ]
    assert metrics[1]["exact_success_count"] == 2
    assert policy_hash == exp.sha256_json(policy)

    heldout = rows + [_candidate("m", "p3", "direct", True, split="heldout")]
    with pytest.raises(exp.CertificationError, match="calibration_rows_only"):
        exp.select_schedule(heldout)

    tied = [
        _candidate("m", "p", schedule, False, split="calibration", parsed=False)
        for schedule in exp.SCHEDULE_ORDER
    ]
    assert exp.select_schedule(tied)[2]["schedule_id"] == "direct"


def test_scenario_verify_6976_heldout_vault_opens_once_after_freeze() -> None:
    """SCENARIO-VERIFY-6976-HELDOUT enforces label-opening chronology."""

    fixture = {
        "exact_witness_rows": [
            {
                "subject_kind": "slice_pair",
                "split": split,
                "pair_id": pair_id,
                "expected_label": "equivalent",
            }
            for split, pair_id in (("calibration", "c"), ("heldout", "h"))
        ]
    }
    vault = exp.SealedLabelVault(fixture)
    assert vault.open("calibration") == {"c": "equivalent"}
    with pytest.raises(exp.CertificationError, match="selected_policy_hash_required"):
        vault.open("heldout")
    assert vault.open("heldout", selected_policy_hash="sha256:frozen") == {"h": "equivalent"}
    with pytest.raises(exp.CertificationError, match="labels_already_opened"):
        vault.open("heldout", selected_policy_hash="sha256:frozen")
    assert vault.opening_rows == [
        {
            "split": "calibration",
            "opening_sequence": 1,
            "selected_policy_hash": None,
            "label_count": 1,
        },
        {
            "split": "heldout",
            "opening_sequence": 2,
            "selected_policy_hash": "sha256:frozen",
            "label_count": 1,
        },
    ]

    with pytest.raises(exp.CertificationError, match="opened_label_missing"):
        exp.apply_labels(
            [_candidate("m", "missing", "direct", False, split="heldout")],
            "heldout",
            {},
        )


def test_scenario_verify_6976_headroom_is_derived_from_group_rows() -> None:
    """SCENARIO-VERIFY-6976-HEADROOM counts mixed groups, not a target."""

    rows = []
    for pair_id, outcomes in (("p1", (True, False, False)), ("p2", (True, True, True))):
        for schedule, success in zip(exp.SCHEDULE_ORDER, outcomes, strict=True):
            rows.append(_candidate("m", pair_id, schedule, success, split="heldout"))
    groups = exp.build_per_group_results(rows, selected_schedule="direct")
    headroom = exp.headroom_rows(groups)
    assert len(groups) == 2
    assert [row["pair_id"] for row in headroom] == ["p1"]
    assert headroom[0]["valid_candidate_count"] == 1
    assert headroom[0]["invalid_candidate_count"] == 2
    assert headroom[0]["within_group_exact_headroom"] == 1
    assert exp.heldout_headroom_group_count(groups) == 1

    broken = rows[:-1]
    with pytest.raises(exp.CertificationError, match="schedule_roster_mismatch"):
        exp.build_per_group_results(broken, selected_schedule="direct")
    wrong_split = deepcopy(rows)
    wrong_split[0]["split"] = "calibration"
    with pytest.raises(exp.CertificationError, match="heldout_rows_only"):
        exp.build_per_group_results(wrong_split, selected_schedule="direct")
    with pytest.raises(exp.CertificationError, match="heldout_rows_only"):
        exp.build_heldout_metric_rows(wrong_split, selected_schedule="direct")
    unopened = deepcopy(rows)
    unopened[0]["exact_semantic_success"] = None
    with pytest.raises(exp.CertificationError, match="heldout_label_not_opened"):
        exp.build_per_group_results(unopened, selected_schedule="direct")


def test_scenario_verify_6976_paired_deltas_use_matching_pair_units() -> None:
    """SCENARIO-VERIFY-6976-PAIRED preserves model rows inside pair resamples."""

    rows = []
    outcomes = {
        ("m1", "p1"): (True, False, False),
        ("m2", "p1"): (True, True, False),
        ("m1", "p2"): (False, False, False),
        ("m2", "p2"): (False, True, False),
    }
    for (model, pair_id), values in outcomes.items():
        for schedule, success in zip(exp.SCHEDULE_ORDER, values, strict=True):
            rows.append(_candidate(model, pair_id, schedule, success, split="heldout"))
    first = exp.paired_schedule_deltas(rows, seed=6976, resamples=200)
    second = exp.paired_schedule_deltas(list(reversed(rows)), seed=6976, resamples=200)
    assert first == second
    assert len(first) == 3
    direct_trigger = next(
        row
        for row in first
        if row["schedule_a"] == "direct" and row["schedule_b"] == "trigger_switched"
    )
    assert direct_trigger["paired_candidate_count"] == 4
    assert direct_trigger["paired_pair_count"] == 2
    assert direct_trigger["wins"] == 1
    assert direct_trigger["losses"] == 1
    assert direct_trigger["ties"] == 2
    assert direct_trigger["mean_delta"] == 0.0
    assert direct_trigger["bootstrap_unit"] == "pair_id"


def test_req_verify_6976_preconditions_report_exact_failures() -> None:
    """SCENARIO-VERIFY-6976-PRECONDITIONS rejects score and hash drift."""

    inputs = exp.load_inputs(REPO)
    checks = exp.check_preconditions(REPO, inputs)
    assert all(row["passed"] for row in checks)

    changed = deepcopy(inputs)
    changed["bank"]["candidate_bank_complete_score"] = 0
    changed["bank"]["schedule_hashes"]["direct"] = "sha256:changed"
    failures = exp.gate_summary(exp.check_preconditions(REPO, changed))
    assert {row["failed_check"] for row in failures} >= {
        "candidate_bank_complete_score",
        "schedule_hashes",
    }

    blocked = exp.build_blocked_artifact(
        date="20260904",
        duration_s=0.1,
        checks=exp.check_preconditions(REPO, changed),
        source_hashes=exp.source_artifact_hashes(REPO),
    )
    exp.validate_artifact(blocked)
    assert blocked["honest_verdict"] == "blocked_exact_candidate_certification"
    assert blocked["candidate_certification_complete_score"] == 0
    assert blocked["selected_policy_ready_score"] == 0
    assert blocked["selected_policy_positive_score"] == 0


def test_req_verify_6976_defensive_boundaries_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6976-PRECONDITIONS covers malformed and absent inputs."""

    nonobject = tmp_path / "not-an-object.json"
    nonobject.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._read_object(nonobject)

    assert exp._raw_roster_observation({"per_attempt_rows": {}, "raw_output_rows": []}) == {
        "attempt_count": 0,
        "raw_count": 0,
        "unique": False,
        "hashes_match": False,
    }
    malformed_roster = exp._raw_roster_observation(
        {"per_attempt_rows": [None], "raw_output_rows": []}
    )
    assert malformed_roster["hashes_match"] is False

    inputs = exp.load_inputs(REPO)
    changed = deepcopy(inputs)
    changed["bank"]["selected_pair_rows"].append(None)
    assert (
        exp._fixture_binding_observation(changed["bank"], changed["fixture"])["all_bindings_match"]
        is False
    )

    unlabeled = [
        {
            **_candidate("m", "p", schedule, False, split="calibration"),
            "exact_semantic_success": None,
        }
        for schedule in exp.SCHEDULE_ORDER
    ]
    with pytest.raises(exp.CertificationError, match="calibration_label_not_opened"):
        exp.select_schedule(unlabeled)

    duplicate_bank = {"selected_pair_rows": [{"pair_id": "p"}, {"pair_id": "p"}]}
    with pytest.raises(exp.CertificationError, match="duplicate_selected_pair_id"):
        exp._pair_index(duplicate_bank)

    bank = deepcopy(inputs["bank"])
    bank["per_attempt_rows"][0]["pair_id"] = "missing"
    with pytest.raises(exp.CertificationError, match="selected_pair_missing"):
        exp.build_certified_artifact(
            bank=bank,
            fixture=inputs["fixture"],
            date="20260904",
            duration_s=0.1,
            checks=[],
            source_hashes={},
        )

    blocked = exp.build_from_paths(tmp_path, date="20260904")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]

    null_hash = exp.sha256_json(None)
    selection_rows = [{"selection_frozen": True} for _ in exp.SCHEDULE_ORDER]
    heldout_rows = [{"terminal": True}]
    assert exp.policy_ready_score(None, null_hash, selection_rows, heldout_rows) == 1
    assert exp.policy_positive_score(None, [], 1, 1) == 0
    assert exp.policy_positive_score({"schedule_id": "direct"}, [], 0, 1) == 0
    assert exp.verdict_for_scores(1, 1, 0, []) == (
        "null",
        "complete_null_exact_candidate_headroom",
    )
    assert exp.verdict_for_scores(0, 0, 0, [{"all_required_agreement": False}]) == (
        "disqualified",
        "complete_disqualified_exact_candidate_certification",
    )
    assert exp.verdict_for_scores(0, 0, 0, [{"all_required_agreement": True}]) == (
        "partial",
        "partial_exact_candidate_certification",
    )


def test_req_verify_6976_frozen_artifact_has_bare_row_derived_results(
    frozen_artifact: dict,
) -> None:
    """SCENARIO-VERIFY-6976-BARE validates the complete frozen finding."""

    artifact = frozen_artifact
    assert len(artifact["per_candidate_rows"]) == 108
    assert len(artifact["parser_outcome_rows"]) == 108
    assert len(artifact["exact_witness_rows"]) == 108
    assert len(artifact["solver_agreement_rows"]) == 108
    assert artifact["selected_policy"]["schedule_id"] == "direct"
    assert artifact["heldout_headroom_group_count"] == len(artifact["heldout_headroom_rows"])
    assert artifact["heldout_headroom_group_count"] == 2
    for field in (
        "candidate_certification_complete_score",
        "selected_policy_ready_score",
        "selected_policy_positive_score",
        "heldout_headroom_group_count",
    ):
        assert type(artifact[field]) is int
    assert artifact["candidate_certification_complete_score"] == 1
    assert artifact["selected_policy_ready_score"] == 1
    assert artifact["selected_policy_positive_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])


def test_req_verify_6976_validator_rejects_headline_and_verdict_drift(
    frozen_artifact: dict,
) -> None:
    """REQ-VERIFY-6976 rejects aggregates that contradict terminal rows."""

    for field, value, reason in (
        ("heldout_headroom_group_count", 99, "heldout_headroom_group_count_mismatch"),
        ("candidate_certification_complete_score", True, "not_bare_int"),
        ("selected_policy_hash", "sha256:changed", "selected_policy_hash_mismatch"),
        ("verdict_class", "positive", "oracle_distinct_positive_forbidden"),
    ):
        changed = deepcopy(frozen_artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.payload_checksum(changed)
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(changed)

    changed = deepcopy(frozen_artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.validate_artifact(changed)


def test_req_verify_6976_validator_rejects_every_derived_surface(
    frozen_artifact: dict,
) -> None:
    """REQ-VERIFY-6976 makes row evidence authoritative over saved summaries."""

    mutations = (
        ("required", lambda row: row.pop("rows"), "required_fields_missing"),
        (
            "principle",
            lambda row: row["field_principles"].__setitem__("rows", ""),
            "field_principles_incomplete",
        ),
        (
            "substrate",
            lambda row: row.__setitem__("inference_substrate", "other"),
            "inference_substrate_mismatch",
        ),
        (
            "oracle",
            lambda row: row.__setitem__("verifier_is_oracle", False),
            "verifier_oracle_declaration_mismatch",
        ),
        ("rows", lambda row: row["rows"].clear(), "rows_projection_mismatch"),
        (
            "count",
            lambda row: (row["rows"].pop(), row["per_candidate_rows"].pop()),
            "candidate_row_count_mismatch",
        ),
        (
            "calibration",
            lambda row: row["calibration_metric_rows"].clear(),
            "calibration_metric_rows_mismatch",
        ),
        (
            "selection",
            lambda row: row["policy_selection_rows"].clear(),
            "policy_selection_rows_mismatch",
        ),
        (
            "heldout_metrics",
            lambda row: row["heldout_metric_rows"].clear(),
            "heldout_metric_rows_mismatch",
        ),
        (
            "groups",
            lambda row: row["per_group_results"].clear(),
            "per_group_results_mismatch",
        ),
        (
            "headroom",
            lambda row: row["heldout_headroom_rows"].clear(),
            "heldout_headroom_rows_mismatch",
        ),
        (
            "paired",
            lambda row: row["paired_schedule_delta_rows"].clear(),
            "paired_schedule_delta_rows_mismatch",
        ),
        (
            "complete",
            lambda row: row.__setitem__("candidate_certification_complete_score", 0),
            "candidate_certification_complete_score_mismatch",
        ),
        (
            "ready",
            lambda row: row.__setitem__("selected_policy_ready_score", 0),
            "selected_policy_ready_score_mismatch",
        ),
        (
            "positive",
            lambda row: row.__setitem__("selected_policy_positive_score", 0),
            "selected_policy_positive_score_mismatch",
        ),
        (
            "verdict",
            lambda row: row.__setitem__("verdict_class", "null"),
            "verdict_class_mismatch",
        ),
        (
            "honest",
            lambda row: row.__setitem__("honest_verdict", "wrong"),
            "honest_verdict_mismatch",
        ),
        (
            "opening",
            lambda row: row["label_opening_rows"].clear(),
            "label_opening_rows_mismatch",
        ),
    )
    for _name, mutate, reason in mutations:
        changed = deepcopy(frozen_artifact)
        mutate(changed)
        changed["reproducibility_checksum"] = exp.payload_checksum(changed)
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(changed)

    blocked = exp.build_blocked_artifact(
        date="20260904",
        duration_s=0.1,
        checks=[exp.gate_check("x", 1, 0)],
        source_hashes={},
    )
    blocked["gate_check_summary"] = []
    blocked["reproducibility_checksum"] = exp.payload_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_artifact_mismatch"):
        exp.validate_artifact(blocked)


def test_req_verify_6976_run_writes_only_the_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_artifact: dict
) -> None:
    """REQ-VERIFY-6976 exposes a validated writer without touching tracked results."""

    monkeypatch.setattr(exp, "build_from_paths", lambda _root, date: deepcopy(frozen_artifact))
    output = tmp_path / "experiment_6976.json"
    written = exp.run(date="20260904", repo_root=REPO, output_path=output)
    assert written == frozen_artifact
    assert json.loads(output.read_text(encoding="utf-8")) == frozen_artifact
