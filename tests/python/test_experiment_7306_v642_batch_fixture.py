"""Tests for the V642 version-bound batch fixture.

Spec refs: REQ-VERIFY-7306 and SCENARIO-VERIFY-7306-*.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_7306_v642_batch_fixture as batch


def _one_group_fixture() -> tuple[dict, dict]:
    public, scorer = batch.build_fixture()
    public["evaluation_groups"] = public["evaluation_groups"][:1]
    wanted = {row["unit_id"] for row in public["evaluation_groups"][0]["claims"]}
    scorer["labels"] = [row for row in scorer["labels"] if row["unit_id"] in wanted]
    return public, scorer


def test_panel_and_call_budget_are_frozen() -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-CALLS."""

    public, scorer = batch.build_fixture()

    assert len(public["development_groups"]) == 8
    assert len(public["evaluation_groups"]) == 16
    assert len(scorer["labels"]) == 24 * 8
    assert "expected_decision" not in batch.canonical_json(public)
    for group in [*public["development_groups"], *public["evaluation_groups"]]:
        assert len(group["source_versions"]) == 2
        assert Counter(row["source_version"] for row in group["claims"]) == {1: 4, 2: 4}
        private = [row for row in scorer["labels"] if row["group_id"] == group["group_id"]]
        for version in (1, 2):
            assert Counter(
                row["case_type"] for row in private if row["source_version"] == version
            ) == {
                "supported": 1,
                "contradicted": 1,
                "unsupported": 1,
                "compositional": 1,
            }

    schedule = batch.build_call_schedule(public["evaluation_groups"])
    assert len(schedule) == 16 * 16
    for group in public["evaluation_groups"]:
        for version in (1, 2):
            rows = [
                row
                for row in schedule
                if row["group_id"] == group["group_id"] and row["source_version"] == version
            ]
            assert Counter(row["arm"] for row in rows) == {
                "serial_versioned_verifier": 5,
                "batched_versioned_verifier": 2,
                "batched_warm_prefix_direct": 1,
            }
            assert {
                arm: sum(row["allocated_output_tokens"] for row in rows if row["arm"] == arm)
                for arm in batch.ARMS
            } == {arm: 1280 for arm in batch.ARMS}
            assert all(len(set(row["claim_ids"])) == len(row["claim_ids"]) for row in rows)
            assert all(row["source_version"] == version for row in rows)


def test_fake_transport_executes_exact_call_multiplicity_and_semantics() -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-CALLS and -E2E."""

    public, scorer = _one_group_fixture()
    transport = batch.CpuFakeTransport()

    execution = batch.execute_public_fixture(public, transport=transport)
    scored = batch.score_predictions(execution["predictions"], scorer["labels"])

    assert len(transport.requests) == 16
    assert Counter(row["arm"] for row in transport.requests) == {
        "serial_versioned_verifier": 10,
        "batched_versioned_verifier": 4,
        "batched_warm_prefix_direct": 2,
    }
    assert len(scored) == 24
    assert all(row["metric"] == 1 for row in scored)
    assert all(row["censored"] is False for row in scored)
    assert all(
        row["actual_output_tokens"] <= row["allocated_output_tokens"]
        for row in execution["call_rows"]
    )
    assert all(row["source_hash"] for row in scored)


def test_joint_results_match_by_unique_id_not_position() -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-IDS."""

    request = {
        "batch_id": "batch-1",
        "source_id": "source-1",
        "source_version": 1,
        "source_hash": "sha256:source",
        "claim_ids": ["c1", "c2", "c3"],
    }
    response = {
        **{key: request[key] for key in ("batch_id", "source_id", "source_version", "source_hash")},
        "items": [
            {"claim_id": "c3", "decision": "unknown"},
            {"claim_id": "c1", "decision": "supported"},
            {"claim_id": "c2", "decision": "contradicted"},
        ],
    }

    parsed = batch.parse_joint_response(request, response, value_key="decision")
    assert [parsed[key]["value"] for key in request["claim_ids"]] == [
        "supported",
        "contradicted",
        "unknown",
    ]

    damaged = deepcopy(response)
    damaged["items"] = [
        {"claim_id": "c3", "decision": "supported"},
        {"claim_id": "c1", "decision": "contradicted"},
        {"claim_id": "c1", "decision": "supported"},
        {"claim_id": "extra", "decision": "supported"},
    ]
    parsed = batch.parse_joint_response(request, damaged, value_key="decision")
    assert parsed["c1"] == {"value": None, "errors": ["duplicate_claim_id"]}
    assert parsed["c2"] == {"value": None, "errors": ["missing_claim_id"]}
    assert parsed["c3"]["value"] == "supported"


def test_source_version_binding_invalidates_and_fails_closed() -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-VERSION."""

    public, _ = _one_group_fixture()
    group = public["evaluation_groups"][0]
    first, second = group["source_versions"]
    compiler = batch.VersionedSourceCompiler()

    v1 = compiler.compile(group["source_id"], first)
    v2 = compiler.compile(group["source_id"], second)
    stale = compiler.compile(group["source_id"], first)
    collision = deepcopy(second)
    collision["document"] = batch.make_document("collision", "Xenia precedes Yarrow.")
    collision["source_hash"] = batch.sha256_bytes(b"Xenia precedes Yarrow.")
    collided = compiler.compile(group["source_id"], collision)

    assert v1["ok"] is True and v1["invalidated_entries"] == 0
    assert v2["ok"] is True and v2["invalidated_entries"] == 1
    assert stale["ok"] is False and stale["error"] == "stale_source_version"
    assert collided["ok"] is False and collided["error"] == "source_id_collision"

    mixed = batch.build_batch_request(
        "batched_versioned_verifier", group, [group["claims"][0], group["claims"][4]]
    )
    assert mixed == {"ok": False, "error": "mixed_source_versions"}


@pytest.mark.parametrize(
    "response,error",
    [
        (None, "malformed_batch_response"),
        ({"items": "bad"}, "batch_identity_mismatch"),
        ({"batch_id": "wrong", "items": []}, "batch_identity_mismatch"),
    ],
)
def test_malformed_output_abstains_all_units(response: object, error: str) -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-IDS."""

    request = {
        "batch_id": "batch-1",
        "source_id": "source-1",
        "source_version": 1,
        "source_hash": "sha256:source",
        "claim_ids": ["c1", "c2"],
    }
    parsed = batch.parse_joint_response(request, response, value_key="decision")

    assert parsed == {
        "c1": {"value": None, "errors": [error]},
        "c2": {"value": None, "errors": [error]},
    }


def test_label_authority_is_separate_and_join_is_strict() -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-AUTHORITY."""

    public, scorer = _one_group_fixture()
    execution = batch.execute_public_fixture(public)
    visible = batch.canonical_json({"public": public, "execution": execution})

    assert "expected_decision" not in visible
    assert "case_type" not in visible
    assert '"labels":' not in visible
    assert scorer["readable_by_prediction_path"] is False

    missing = scorer["labels"][:-1]
    with pytest.raises(ValueError, match="label_identity_mismatch"):
        batch.score_predictions(execution["predictions"], missing)
    duplicate = [*scorer["labels"], scorer["labels"][0]]
    with pytest.raises(ValueError, match="label_identity_mismatch"):
        batch.score_predictions(execution["predictions"], duplicate)


def test_adverse_controls_prevent_unrelated_contamination() -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-ISOLATION."""

    public, _ = _one_group_fixture()
    controls = batch.run_batch_controls(public)
    by_name = {row["control"]: row for row in controls}

    assert set(by_name) == set(batch.REQUIRED_CONTROLS)
    assert all(row["passed"] is True for row in controls)
    assert by_name["unrelated_claim_contamination"]["observed"] == 0
    assert by_name["claim_order_permutation"]["observed"] is True
    assert by_name["stale_source_substitution"]["observed"] == "rejected"
    assert by_name["malformed_output"]["observed"] == "all_abstain"


def test_group_bootstrap_and_gates_keep_readiness_separate_from_value() -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-GATES."""

    public, scorer = batch.build_fixture()
    execution = batch.execute_public_fixture(public)
    rows = batch.score_predictions(execution["predictions"], scorer["labels"])
    bootstrap = batch.paired_group_bootstrap(rows, draws=10_000, seed=batch.BOOTSTRAP_SEED)
    controls = batch.run_batch_controls(public)
    gates = batch.acceptance_gates(rows, controls, bootstrap, execution["call_rows"])
    by_name = {row["criterion"]: row for row in gates}

    assert bootstrap["draw_count"] == 10_000
    assert bootstrap["independent_group_count"] == 16
    assert bootstrap["accuracy_difference"]["one_sided_95_lower"] == 0.0
    assert by_name["protocol_and_controls"]["passed"] is True
    assert by_name["serial_batch_semantic_parity"]["passed"] is True
    assert by_name["full_cost_speedup_lower_vs_serial"]["observed"] is None
    assert by_name["full_cost_speedup_lower_vs_serial"]["passed"] is False
    assert by_name["full_cost_speedup_lower_vs_direct"]["observed"] is None

    artifact = batch.assemble_artifact(
        public,
        scorer,
        execution,
        rows,
        controls,
        bootstrap,
        gates,
        validation_receipts=batch.validation_receipt_fixture(),
    )
    assert artifact["batch_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert batch.validate_artifact(artifact) == []


def test_precondition_failure_preserves_exact_gate_summary(tmp_path) -> None:
    """REQ-VERIFY-7306 records exact blocked dependency failures."""

    checks = batch.authenticate_inputs(
        tmp_path,
        overrides={name: tmp_path / f"missing-{name}" for name in batch.SOURCE_PATHS},
    )
    summary = batch.gate_check_summary(checks)

    assert summary == {
        "failed_check": "required_path_available",
        "upstream": "agents",
        "field": "path",
        "expected_value": "file",
        "observed_value": "missing",
    }


def test_artifact_validator_rejects_schema_and_checksum_changes() -> None:
    """REQ-VERIFY-7306 keeps the terminal artifact cold-valid."""

    public, scorer = batch.build_fixture()
    execution = batch.execute_public_fixture(public)
    rows = batch.score_predictions(execution["predictions"], scorer["labels"])
    controls = batch.run_batch_controls(public)
    bootstrap = batch.paired_group_bootstrap(rows, draws=10_000, seed=batch.BOOTSTRAP_SEED)
    gates = batch.acceptance_gates(rows, controls, bootstrap, execution["call_rows"])
    artifact = batch.assemble_artifact(
        public,
        scorer,
        execution,
        rows,
        controls,
        bootstrap,
        gates,
        validation_receipts=batch.validation_receipt_fixture(),
    )
    damaged = deepcopy(artifact)
    damaged["MODEL_SPECS"] = [{"model": "not-invoked"}]

    errors = batch.validate_artifact(damaged)
    assert "MODEL_SPECS" in errors
    assert "reproducibility_checksum" in errors


def test_fail_closed_parser_and_schedule_boundaries(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-VERIFY-7306; SCENARIO-VERIFY-7306-IDS and -VERSION."""

    public, _ = _one_group_fixture()
    group = public["evaluation_groups"][0]
    batch._progress(9, "test", "visible")
    assert "[exp7306] phase 9 test: visible" in capsys.readouterr().out

    with pytest.raises(ValueError, match="relation_mention_identity"):
        batch._mention_at({"mentions": []}, 0)
    assert batch._completion(batch.make_document("plain", "no relation here"), source=True) == {
        "outcome": "unknown",
        "relations": [],
    }

    damaged_sources = deepcopy(group)
    damaged_sources["source_versions"] = []
    with pytest.raises(ValueError, match="source_version_identity"):
        batch._source_for(damaged_sources, 1)
    damaged_claims = deepcopy(group)
    damaged_claims["claims"] = damaged_claims["claims"][:-1]
    with pytest.raises(ValueError, match="claim_version_denominator"):
        batch._claims_for(damaged_claims, 2)
    with pytest.raises(ValueError, match="evaluation_group_denominator"):
        batch.build_call_schedule([group])

    request = {
        "batch_id": "batch-1",
        "source_id": "source-1",
        "source_version": 1,
        "source_hash": "sha256:source",
        "claim_ids": ["c1"],
    }
    malformed_items = {**request, "items": "bad"}
    assert batch.parse_joint_response(request, malformed_items, value_key="decision")["c1"][
        "errors"
    ] == ["malformed_batch_response"]
    missing_value = {**request, "items": [{"claim_id": "c1"}, "ignored"]}
    assert batch.parse_joint_response(request, missing_value, value_key="decision")["c1"][
        "errors"
    ] == ["malformed_item"]

    compiler = batch.VersionedSourceCompiler()
    assert compiler.compile("source", {"source_version": True})["error"] == (
        "invalid_source_provenance"
    )
    source = batch._source_for(group, 1)
    bad_hash = {**source, "source_hash": "sha256:wrong"}
    assert compiler.compile("source", bad_hash)["error"] == "source_hash_mismatch"

    invalid_request = {
        "batch_id": "batch",
        "source_id": "source",
        "source_version": 1,
        "source_hash": "sha256:source",
        "call_type": "invalid",
    }
    with pytest.raises(ValueError, match="call_type"):
        batch.CpuFakeTransport().call(invalid_request)


def test_malformed_transport_responses_abstain_without_repairs() -> None:
    """REQ-VERIFY-7306; malformed source, serial claim, and joint output fail closed."""

    public, _ = _one_group_fixture()

    def mutate(request: dict, response: dict) -> object:
        if (
            request["arm"] == "serial_versioned_verifier"
            and request["source_version"] == 1
            and request["call_type"] == "source"
        ):
            return None
        if (
            request["arm"] == "serial_versioned_verifier"
            and request["source_version"] == 2
            and request["call_type"] == "claim"
        ):
            return None
        if (
            request["arm"] == "batched_versioned_verifier"
            and request["source_version"] == 1
            and request["call_type"] == "claim_batch"
        ):
            response["items"] = response["items"][1:]
        return response

    execution = batch.execute_public_fixture(public, transport=batch.CpuFakeTransport(mutate))
    rows = execution["predictions"]

    serial_v1 = [
        row
        for row in rows
        if row["arm"] == "serial_versioned_verifier" and row["source_version"] == 1
    ]
    serial_v2 = [
        row
        for row in rows
        if row["arm"] == "serial_versioned_verifier" and row["source_version"] == 2
    ]
    batched_v1 = [
        row
        for row in rows
        if row["arm"] == "batched_versioned_verifier" and row["source_version"] == 1
    ]
    assert all(row["errors"] == ["source_compile_failed"] for row in serial_v1)
    assert all(row["errors"] == ["malformed_claim_response"] for row in serial_v2)
    assert sum(row["errors"] == ["missing_claim_id"] for row in batched_v1) == 1


def test_blocked_artifact_and_available_source_receipts(tmp_path) -> None:
    """REQ-VERIFY-7306 preserves exact preconditions and historical provenance."""

    for name in batch.SOURCE_PATHS:
        (tmp_path / name).write_text(name, encoding="utf-8")
    overrides = {name: tmp_path / name for name in batch.SOURCE_PATHS}
    checks = batch.authenticate_inputs(tmp_path, overrides=overrides)
    assert all(row["passed"] for row in checks)
    hashed = [row for row in checks if "sha256" in row]
    assert len(hashed) == len(batch.SOURCE_PATHS)
    assert all(row["sha256"].startswith("sha256:") for row in hashed)
    assert batch.gate_check_summary(checks) == {
        "failed_check": None,
        "upstream": None,
        "field": None,
        "expected_value": None,
        "observed_value": None,
    }

    failed = deepcopy(checks)
    failed[0]["passed"] = False
    failed[0]["observed_value"] = "missing"
    blocked = batch.blocked_artifact(batch.RUN_DATE, failed, 0.25)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["batch_fixture_ready_score"] == 0
    assert blocked["gate_check_summary"]["observed_value"] == "missing"
    assert blocked["reproducibility_checksum"] == batch.artifact_checksum(blocked)

    root = Path(__file__).resolve().parents[2]
    real_checks = batch.authenticate_inputs(root)
    hashes = batch._historical_source_hashes(root, real_checks)
    audit = hashes[str(batch.SOURCE_PATHS["v641_audit_artifact"])]
    assert audit["producer_identity"] == "exp7294-reuse-audit"
    assert audit["terminal_class"] == "disqualified"
    assert audit["historical_evidence_only"] is True
    assert audit["dependency_gate"] is False


def test_validator_reports_each_terminal_contract_failure() -> None:
    """REQ-VERIFY-7306 cold validation rejects malformed terminal records."""

    public, scorer = batch.build_fixture()
    execution = batch.execute_public_fixture(public)
    rows = batch.score_predictions(execution["predictions"], scorer["labels"])
    controls = batch.run_batch_controls(public)
    bootstrap = batch.paired_group_bootstrap(rows, draws=10_000, seed=batch.BOOTSTRAP_SEED)
    gates = batch.acceptance_gates(rows, controls, bootstrap, execution["call_rows"])
    artifact = batch.assemble_artifact(
        public,
        scorer,
        execution,
        rows,
        controls,
        bootstrap,
        gates,
        validation_receipts=batch.validation_receipt_fixture(),
    )

    damaged = deepcopy(artifact)
    del damaged["schema"]
    damaged.update(
        {
            "run_date": "wrong",
            "rows": [],
            "batch_control_rows": [],
            "batch_fixture_ready_score": 0,
            "verdict_class": "positive",
            "honest_verdict": "wrong",
        }
    )
    errors = batch.validate_artifact(damaged)
    assert {
        "schema",
        "run_date",
        "rows",
        "batch_control_rows",
        "batch_fixture_ready_score",
        "verdict_class",
        "honest_verdict",
        "reproducibility_checksum",
    }.issubset(errors)

    wrong_arms = deepcopy(artifact)
    wrong_arms["rows"][0]["arm"] = "wrong"
    wrong_arms["reproducibility_checksum"] = batch.artifact_checksum(wrong_arms)
    assert "row_arm_denominator" in batch.validate_artifact(wrong_arms)

    with pytest.raises(Exception, match=f"date must be {batch.RUN_DATE}"):
        batch._date_argument("20260101")
    assert batch._date_argument(batch.RUN_DATE) == batch.RUN_DATE
