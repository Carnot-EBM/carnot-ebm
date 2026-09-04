"""Tests for the certified mapping-error and unused-pair fixture.

Spec refs: REQ-VERIFY-6967 and SCENARIO-VERIFY-6967-*.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6967_certified_error_headroom_fixture as exp


ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict[str, object]:
    """Load one frozen predecessor for a test that must use real bytes."""

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predecessor_inputs() -> dict[str, object]:
    """Load all inputs once so tests do not rewrite the research record."""

    return exp.load_inputs(ROOT)


@pytest.fixture(scope="module")
def artifact(predecessor_inputs: dict[str, object]) -> dict[str, object]:
    """Build the complete exact fixture once for all integration assertions."""

    return exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        inputs=predecessor_inputs,
    )


def test_req_verify_6967_spec_precedes_code_and_declares_contract() -> None:
    """REQ-VERIFY-6967 declares every artifact field and failure scenario."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6967") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-VERIFY-6967-{name}" in section
        for name in (
            "PRECONDITIONS",
            "RECOMPUTATION",
            "CLUSTERS",
            "EXCLUSION",
            "BALANCE",
            "DISJOINTNESS",
            "SEALING",
            "CHRONOLOGY",
            "HEADROOM",
            "GATES",
        )
    )


def test_scenario_verify_6967_recomputation_reparses_raw_bytes(
    predecessor_inputs: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6967-RECOMPUTATION ignores a saved parse decision."""

    bank = predecessor_inputs["bank"]
    pairs = predecessor_inputs["fixture_pairs"]
    malformed = deepcopy(bank["attempt_rows"][0])
    malformed["parse"] = {
        "json_valid": True,
        "schema_valid": True,
        "parsed_candidate": {"mapping": {}},
    }

    row, witness = exp.recompute_proposal(malformed, pairs[malformed["pair_id"]])

    assert row["parse_failure"] is True
    assert row["parse_reason"] == "malformed_json"
    assert row["error_signature"] == ["parse:malformed_json"]
    assert witness["enumeration_status"] == "parse_rejected"
    assert witness["z3_status"] == "parse_rejected"


def test_scenario_verify_6967_clusters_are_deterministic_and_exclusive() -> None:
    """SCENARIO-VERIFY-6967-CLUSTERS assigns each error to one stable group."""

    rows = [
        {
            "attempt_key": "b",
            "exact_mapping_correct": False,
            "error_signature": ["schema:source_domain_roster"],
            "repair_stage": "schema_repair",
            "model_family": "model_b",
            "formulation_family": "family_a",
            "prompt_variant_id": "direct_affine",
            "raw_sha256": "sha256:b",
            "exact_evidence_hash": "sha256:eb",
        },
        {
            "attempt_key": "a",
            "exact_mapping_correct": False,
            "error_signature": ["schema:source_domain_roster"],
            "repair_stage": "schema_repair",
            "model_family": "model_a",
            "formulation_family": "family_b",
            "prompt_variant_id": "domain_first",
            "raw_sha256": "sha256:a",
            "exact_evidence_hash": "sha256:ea",
        },
        {
            "attempt_key": "ok",
            "exact_mapping_correct": True,
            "error_signature": [],
            "repair_stage": None,
            "model_family": "model_a",
            "formulation_family": "family_a",
            "prompt_variant_id": "objective_first",
            "raw_sha256": "sha256:ok",
            "exact_evidence_hash": "sha256:eok",
        },
    ]

    forward = exp.cluster_error_rows(rows)
    reverse = exp.cluster_error_rows(list(reversed(rows)))

    assert forward == reverse
    assert len(forward["error_cluster_rows"]) == 1
    assert forward["error_cluster_rows"][0]["count"] == 2
    assert forward["error_cluster_rows"][0]["model_family_counts"] == {
        "model_a": 1,
        "model_b": 1,
    }
    assert [row["attempt_key"] for row in forward["cluster_example_rows"]] == ["a", "b"]


def test_scenarios_verify_6967_exclusion_balance_and_disjointness(
    predecessor_inputs: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6967-EXCLUSION, BALANCE, and DISJOINTNESS freeze clean sets."""

    bank = predecessor_inputs["bank"]
    pairs = list(predecessor_inputs["fixture_pairs"].values())
    used = exp.used_pair_ids(bank)
    first = exp.freeze_slices(pairs, used, seed=exp.RANDOM_SEED)
    second = exp.freeze_slices(list(reversed(pairs)), used, seed=exp.RANDOM_SEED)

    assert first == second
    assert len(first["calibration"]) == 18
    assert len(first["heldout"]) == 18
    assert len(first["chronological"]) == 24
    split_ids = {name: {row["pair_id"] for row in rows} for name, rows in first.items()}
    assert not used & set().union(*split_ids.values())
    assert all(len(left & right) == 0 for left, right in exp.pairwise_sets(split_ids).values())

    for name, rows in first.items():
        expected_family_count = 8 if name == "chronological" else 6
        assert Counter(row["family"] for row in rows) == {
            family: expected_family_count for family in exp.FAMILIES
        }
        for family in exp.FAMILIES:
            labels = {row["expected_label"] for row in rows if row["family"] == family}
            assert labels == {"equivalent", "non_equivalent"}


def test_scenario_verify_6967_preconditions_fail_closed_with_complete_schema(
    predecessor_inputs: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6967-PRECONDITIONS gives a diagnostic blocked artifact."""

    broken = deepcopy(predecessor_inputs)
    broken["bank"]["raw_output_rows"] = broken["bank"]["raw_output_rows"][:-1]
    artifact = exp.build_artifact(date="20260904", repo_root=ROOT, inputs=broken)

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_certified_error_headroom_fixture"
    assert artifact["error_fixture_ready_score"] == 0
    assert artifact["chronological_event_stream_ready_score"] == 0
    assert artifact["gate_check_summary"]
    assert all(
        {"failed_check", "expected_value", "observed_value"} <= set(row)
        for row in artifact["gate_check_summary"]
    )


def test_req_verify_6967_complete_artifact_has_sealed_balanced_slices(
    artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6967 keeps public prompts separate from exact labels."""

    assert len(artifact["recomputed_proposal_rows"]) == 162
    assert len(artifact["calibration_rows"]) == 18
    assert len(artifact["heldout_rows"]) == 18
    assert len(artifact["chronological_event_rows"]) == 24
    assert len(artifact["prompt_visible_rows"]) == 60
    assert all(row["authorities_agree"] is True for row in artifact["solver_agreement_rows"])
    assert all(row["disjoint"] is True for row in artifact["split_disjointness_rows"])
    assert all(row["balance_passed"] is True for row in artifact["family_balance_rows"])
    assert all(row["model_headroom_claim"] is None for row in artifact["headroom_opportunity_rows"])
    assert all(row["live_candidate_count"] == 0 for row in artifact["headroom_opportunity_rows"])

    forbidden = exp.find_forbidden_prompt_paths(artifact["prompt_visible_rows"])
    assert forbidden == []
    assert set(artifact["sealed_label_hashes"]) == {
        "calibration",
        "heldout",
        "chronological",
    }


def test_scenario_verify_6967_chronology_and_bare_gates(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6967-CHRONOLOGY and GATES preserve order and bare scores."""

    events = artifact["chronological_event_rows"]
    for index, row in enumerate(events):
        expected_predecessor = None if index == 0 else events[index - 1]["event_id"]
        assert row["event_ordinal"] == index
        assert row["predecessor_event_id"] == expected_predecessor
        assert row["dependency_ids"]
        assert row["dependency_record_hash"].startswith("sha256:")
        assert row["source_certificate_hash"].startswith("sha256:")
        assert row["event_hash"].startswith("sha256:")
        assert "outcome" not in row

    assert type(artifact["error_fixture_ready_score"]) is int
    assert type(artifact["chronological_event_stream_ready_score"]) is int
    assert artifact["error_fixture_ready_score"] == 1
    assert artifact["chronological_event_stream_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")


def test_req_verify_6967_validation_rejects_row_and_checksum_drift(
    artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6967 derives terminal claims from rows and stable hashes."""

    exp.validate_artifact(artifact)
    changed = deepcopy(artifact)
    changed["recomputed_proposal_rows"].pop()
    with pytest.raises(ValueError, match="recomputed_proposal_count"):
        exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(changed)


def test_req_verify_6967_predecessors_have_expected_observed_findings() -> None:
    """REQ-VERIFY-6967 anchors diagnosis to the frozen V609 result, not a claim."""

    certificate = _json(ROOT / exp.SOURCE_PATHS["certificate"])
    selection = _json(ROOT / exp.SOURCE_PATHS["selection"])
    overall = certificate["overall_metric_rows"][0]

    assert overall["proposal_count"] == 162
    assert overall["exact_mapping_correct_count"] == 10
    assert overall["parse_failure_count"] == 79
    assert overall["schema_failure_count"] == 37
    assert {row["available_headroom"] for row in selection["headroom_rows"]} == {0}


def test_req_verify_6967_defensive_helper_boundaries(
    tmp_path: Path,
    predecessor_inputs: dict[str, object],
) -> None:
    """REQ-VERIFY-6967 covers malformed inputs and non-model terminal calls."""

    output = tmp_path / "nested/result.json"
    exp.write_json_atomic(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}

    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._read_object(not_object)

    assert exp._raw_roster_observation({"attempt_rows": {}, "raw_output_rows": []}) == {
        "attempt_count": 0,
        "raw_count": 0,
        "unique": False,
        "hashes_match": False,
    }
    malformed_roster = exp._raw_roster_observation({"attempt_rows": [None], "raw_output_rows": []})
    assert malformed_roster["hashes_match"] is False

    bank = predecessor_inputs["bank"]
    pairs = predecessor_inputs["fixture_pairs"]
    failed_call = deepcopy(bank["attempt_rows"][0])
    failed_call["call_status"] = "timeout"
    failed_call["failure_reason"] = "deadline"
    rebuilt = exp._attempt_for_recomputation(failed_call, pairs[failed_call["pair_id"]])
    assert rebuilt["parse"]["failure_reason"] == "deadline"

    assert exp._prior_certificate_match({}, None) is False
    with pytest.raises(ValueError, match="insufficient_unused_pairs"):
        exp.freeze_slices([], set(), seed=exp.RANDOM_SEED)
    assert exp.find_forbidden_prompt_paths([{"nested": {"expected_label": "hidden"}}]) == [
        "[0].nested.expected_label"
    ]
    assert exp._chronology_ready([]) is False


def test_scenario_verify_6967_error_signature_covers_solver_and_relation_paths() -> None:
    """SCENARIO-VERIFY-6967-CLUSTERS retains solver failures and pure label mismatch."""

    bundle = {
        "proposal_row": {
            "exact_mapping_correct": False,
            "timeout": True,
            "unknown": True,
            "certified_relation": None,
            "canonical_relation": "equivalent",
        },
        "parse_row": {"parse_failure": False},
        "schema_row": {"schema_valid": True},
        "authority_agreement_row": {"authorities_agree": False},
        "enumeration_row": {},
        "z3_row": {},
    }
    assert exp._error_signature(bundle) == [
        "solver:timeout",
        "solver:unknown",
        "solver:authority_disagreement",
    ]

    bundle["proposal_row"].update({"timeout": False, "unknown": False})
    bundle["authority_agreement_row"]["authorities_agree"] = True
    bundle["proposal_row"]["certified_relation"] = "non_equivalent"
    assert exp._error_signature(bundle) == ["relation_mismatch:non_equivalent_vs_equivalent"]


def test_req_verify_6967_validation_guards_each_terminal_claim(
    artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6967 rejects schema, gate, verdict, and headroom drift."""

    mutations = [
        (lambda row: row.pop("rows"), "required_artifact_fields"),
        (lambda row: row["field_principles"].pop("rows"), "field_principles"),
        (lambda row: row.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda row: row.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda row: row.update({"error_fixture_ready_score": True}), "bare_gate_field"),
        (lambda row: row.update({"error_fixture_ready_score": 0}), "error_fixture_ready_score"),
        (
            lambda row: row.update({"chronological_event_stream_ready_score": 0}),
            "chronological_event_stream_ready_score",
        ),
        (lambda row: row.update({"verdict_class": "null"}), "verdict_class"),
        (
            lambda row: row["headroom_opportunity_rows"][0].update({"model_headroom_claim": 0.5}),
            "fabricated_headroom",
        ),
    ]
    for mutate, reason in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(changed)

    blocked = exp.build_blocked_artifact(
        date="20260904",
        duration_s=0.0,
        checks=[exp.gate_check("input", True, False)],
        source_hashes={},
    )
    blocked["gate_check_summary"] = []
    with pytest.raises(ValueError, match="blocked_gate_check_summary"):
        exp.validate_artifact(blocked)

    broken_events = deepcopy(artifact["chronological_event_rows"])
    broken_events[0]["event_hash"] = "sha256:wrong"
    assert exp._chronology_ready(broken_events) is False


def test_req_verify_6967_run_and_cli_surfaces(
    tmp_path: Path,
    artifact: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6967 writes valid and load-blocked artifacts through public surfaces."""

    valid_output = tmp_path / "valid.json"
    monkeypatch.setattr(exp, "build_artifact", lambda **_kwargs: deepcopy(artifact))
    written = exp.run(date="20260904", repo_root=ROOT, output_path=valid_output)
    assert written["error_fixture_ready_score"] == 1
    assert json.loads(valid_output.read_text(encoding="utf-8"))["experiment_id"] == 6967

    blocked_output = tmp_path / "blocked.json"
    blocked = exp.run(date="20260904", repo_root=tmp_path / "missing", output_path=blocked_output)
    assert blocked["verdict_class"] == "blocked"
    assert json.loads(blocked_output.read_text(encoding="utf-8"))["gate_check_summary"]

    monkeypatch.setattr(exp, "run", lambda **_kwargs: deepcopy(artifact))
    assert (
        exp.main(
            [
                "--date",
                "20260904",
                "--repo-root",
                str(ROOT),
                "--output",
                str(tmp_path / "cli.json"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["error_fixture_ready_score"] == 1
