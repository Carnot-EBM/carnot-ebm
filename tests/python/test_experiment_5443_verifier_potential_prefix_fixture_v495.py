"""Tests for Exp5443 verifier-potential prefix fixture.

Spec refs: REQ-SAFE-5443, SCENARIO-SAFE-5443.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5443_verifier_potential_prefix_fixture_v495 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5443_verifier_potential_prefix_fixture_v495.py -q"
)


def _artifact() -> dict[str, Any]:
    return mod.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])


def test_req_safe_5443_spec_declares_verifier_potential_contract() -> None:
    """REQ-SAFE-5443: OpenSpec anchors the V495 prefix-potential fixture."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5443") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5443",
        "SCENARIO-SAFE-5443",
        str(mod.RESULT_RELATIVE_PATH),
        "schema-only traps",
        "semantic contradictions",
        "unreachable tool actions",
        "arithmetic/finite-domain constraints",
        "ontology/triple updates",
        "API-call witnesses",
        "benign rows",
        "`deterministic_verifier_fixture_no_llm`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_safe_5443_builds_complete_fixture_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAFE-5443: complete fixture records deterministic prefix evidence."""

    out_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=out_path, tests_run=[TEST_COMMAND], write=True)
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    metrics = mod.derive_metrics(artifact["fixture_rows"])
    budget = mod.derive_reward_budget(artifact["fixture_rows"])

    assert reloaded == artifact
    mod.validate_artifact(artifact)
    assert artifact["fixture_count"] == len(artifact["fixture_rows"])
    assert artifact["fixture_count"] == metrics["fixture_count"]
    assert artifact["constraint_family_counts"] == metrics["constraint_family_counts"]
    assert set(mod.REQUIRED_CONSTRAINT_FAMILIES).issubset(set(artifact["constraint_family_counts"]))
    assert artifact["prefix_final_disagreement_cases"] == metrics["prefix_final_disagreement_cases"]
    assert artifact["prefix_final_disagreement_cases"] > 0
    assert artifact["exact_final_authority"] is True
    assert artifact["metric_independence_checks_passed"] is True
    assert artifact["verifier_potential_fixture_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reward_evaluation_budget"] == budget
    assert artifact["row_checksums"] == [row["row_checksum"] for row in artifact["fixture_rows"]]
    assert artifact["fixture_checksums"] == [
        row["fixture_checksum"] for row in artifact["fixture_rows"]
    ]

    potential_ids = {row["function_id"] for row in artifact["prefix_potential_functions"]}
    assert potential_ids == {row["function_id"] for row in mod.PREFIX_POTENTIAL_FUNCTIONS}
    assert any(row["monotone"] for row in artifact["prefix_potential_functions"])
    assert all(
        "abstain" in row["unknown_prefix_policy"] for row in artifact["prefix_potential_functions"]
    )

    exact_verifier_rows = [
        row for row in artifact["fixture_rows"] if row["exact_final_verdict"]["verified"]
    ]
    assert len(exact_verifier_rows) == artifact["fixture_count"]
    assert any(not row["exact_final_verdict"]["accepted"] for row in exact_verifier_rows)
    assert any(row["exact_final_verdict"]["accepted"] for row in exact_verifier_rows)

    unknown_prefixes = [
        prefix
        for row in artifact["fixture_rows"]
        for prefix in row["prefixes"]
        if prefix["prefix_id"].endswith(":empty")
    ]
    assert unknown_prefixes
    for prefix in unknown_prefixes:
        assert prefix["accepted_by_potential"] is False
        assert {evaluation["decision"] for evaluation in prefix["potential_evaluations"]} == {
            "abstain"
        }
        assert {evaluation["score"] for evaluation in prefix["potential_evaluations"]} == {0.0}


def test_req_safe_5443_prefix_final_separation_and_exact_final_authority() -> None:
    """REQ-SAFE-5443: accepted prefixes cannot override exact final rejection."""

    artifact = _artifact()
    disagreement_rows = [
        row
        for row in artifact["fixture_rows"]
        if not row["exact_final_verdict"]["accepted"]
        and any(prefix["accepted_by_potential"] for prefix in row["prefixes"])
    ]

    assert disagreement_rows
    for row in disagreement_rows:
        accepted_prefixes = [
            prefix for prefix in row["prefixes"] if prefix["accepted_by_potential"]
        ]
        assert accepted_prefixes
        assert row["exact_final_verdict"]["authority"] == "exact_final_verifier"
        assert row["exact_final_verdict"]["overrides_prefix_potential"] is True
        assert row["accepted_by_final_authority"] is False

    tampered_row = deepcopy(artifact)
    tampered_row["fixture_rows"][0]["exact_final_verdict"]["accepted"] = True
    tampered_row["fixture_rows"][0]["accepted_by_final_authority"] = True
    with pytest.raises(ValueError, match="exact final verdict"):
        mod.validate_artifact(tampered_row)

    tampered_authority = deepcopy(artifact)
    tampered_authority["exact_final_authority"] = False
    with pytest.raises(ValueError, match="exact_final_authority"):
        mod.validate_artifact(tampered_authority)


def test_req_safe_5443_metric_independence_rejects_aggregate_drift() -> None:
    """REQ-SAFE-5443: copied aggregate metrics fail row recomputation."""

    artifact = _artifact()
    metrics = artifact["metric_details"]

    assert metrics["prefix_final_disagreement_row_ids"]
    assert metrics["prefix_final_disagreement_row_ids"] != metrics["final_rejected_row_ids"]
    assert (
        metrics["predicate_support"]["prefix_final_disagreement"]
        != metrics["predicate_support"]["final_rejected"]
    )

    bad_count = deepcopy(artifact)
    bad_count["prefix_final_disagreement_cases"] = len(metrics["final_rejected_row_ids"])
    with pytest.raises(ValueError, match="prefix_final_disagreement_cases"):
        mod.validate_artifact(bad_count)

    bad_ready = deepcopy(artifact)
    bad_ready["metric_independence_checks_passed"] = False
    with pytest.raises(ValueError, match="metric_independence_checks_passed"):
        mod.validate_artifact(bad_ready)

    bad_family = deepcopy(artifact)
    bad_family["constraint_family_counts"] = {"benign": artifact["fixture_count"]}
    with pytest.raises(ValueError, match="constraint_family_counts"):
        mod.validate_artifact(bad_family)


def test_req_safe_5443_budget_checksums_and_potential_specs_fail_closed() -> None:
    """REQ-SAFE-5443: budgets, checksums, and potential definitions are validated."""

    artifact = _artifact()

    bad_budget = deepcopy(artifact)
    bad_budget["reward_evaluation_budget"]["total_cost_units"] += 1
    with pytest.raises(ValueError, match="reward_evaluation_budget"):
        mod.validate_artifact(bad_budget)

    bad_row_budget = deepcopy(artifact)
    bad_row_budget["fixture_rows"][0]["reward_evaluation_budget"]["total_cost_units"] += 1
    with pytest.raises(ValueError, match="reward_evaluation_budget row entry"):
        mod.validate_artifact(bad_row_budget)

    bad_row_checksum = deepcopy(artifact)
    bad_row_checksum["fixture_rows"][0]["row_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="row_checksums"):
        mod.validate_artifact(bad_row_checksum)

    bad_fixture_checksum = deepcopy(artifact)
    bad_fixture_checksum["fixture_rows"][0]["fixture_checksum"] = "1" * 64
    with pytest.raises(ValueError, match="fixture_checksums"):
        mod.validate_artifact(bad_fixture_checksum)

    bad_function = deepcopy(artifact)
    bad_function["prefix_potential_functions"][1]["monotone"] = True
    with pytest.raises(ValueError, match="prefix_potential_functions"):
        mod.validate_artifact(bad_function)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)


def test_req_safe_5443_validation_and_exact_verifier_edge_branches() -> None:
    """REQ-SAFE-5443: defensive validation and exact-check branches stay covered."""

    artifact = _artifact()
    validation_cases: list[tuple[str, Any, str]] = [
        ("field_principles", {}, "field_principles"),
        ("fixture_rows", "bad", "fixture_rows"),
        ("fixture_count", -1, "fixture_count"),
        ("constraint_family_counts", [], "constraint_family_counts"),
        ("prefix_final_disagreement_cases", -1, "prefix_final_disagreement_cases"),
        ("reward_evaluation_budget", [], "reward_evaluation_budget"),
        ("row_provenance_checksum", "bad", "row_provenance_checksum"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("metric_independence_checks_passed", "yes", "metric_independence_checks_passed"),
        ("honest_verdict", "done", "honest_verdict"),
        ("research_conductor_modified", True, "research_conductor.py"),
    ]
    for field, value, expected in validation_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    del missing["fixture_count"]
    assert "missing required fields" in "; ".join(mod.artifact_schema_errors(missing))

    no_disagreement = deepcopy(artifact)
    no_disagreement["prefix_final_disagreement_cases"] = 0
    with pytest.raises(ValueError, match="disagreement"):
        mod.validate_artifact(no_disagreement)

    sparse_metrics = {"constraint_family_counts": {"benign": 1}}
    assert mod._honest_verdict(False, sparse_metrics).startswith("blocked: missing")
    full_metrics = {
        "constraint_family_counts": {name: 1 for name in mod.REQUIRED_CONSTRAINT_FAMILIES}
    }
    assert mod._honest_verdict(False, full_metrics) == (
        "blocked: verifier-potential fixture readiness checks failed"
    )
    assert mod.run(write=False)["verifier_potential_fixture_ready"] is True

    rows = {row["constraint_family"]: row for row in artifact["fixture_rows"]}
    semantic_score = mod._semantic_pair_potential({"object": "open", "negated_object": "closed"})
    assert semantic_score[1] == "accept"
    assert mod._semantic_pair_potential({"object": "open", "negated_object": "open"})[1] == "reject"

    action_row = rows["unreachable_tool_action"]
    assert mod._action_exact_reason(action_row, {"tool": "void_order"}) == "tool_not_allowed"
    assert (
        mod._action_exact_reason(
            action_row,
            {"tool": "cancel_order", "order_state": "draft", "lock_active": False},
        )
        == "order_state_unreachable"
    )
    assert (
        mod._action_exact_reason(
            action_row,
            {"tool": "cancel_order", "order_state": "paid", "lock_active": False},
        )
        is None
    )

    arithmetic_row = rows["arithmetic_finite_domain"]
    assert (
        mod._arithmetic_exact_reason(arithmetic_row, {"x": 10, "y": 1, "sum": 11})
        == "finite_domain_failed"
    )
    assert mod._arithmetic_exact_reason(arithmetic_row, {"x": 2, "y": 3, "sum": 5}) is None

    ontology_row = rows["ontology_triple_update"]
    assert (
        mod._ontology_exact_reason(
            ontology_row,
            {"subject": "bolt-7", "predicate": "attached_to", "object": "assembly-1"},
        )
        == "ontology_predicate_failed"
    )
    assert (
        mod._ontology_exact_reason(
            ontology_row,
            {"subject": "bolt-7", "predicate": "part_of", "object": "bolt-7"},
        )
        == "ontology_object_type_failed"
    )
    assert (
        mod._ontology_exact_reason(
            ontology_row,
            {"subject": "bolt-7", "predicate": "part_of", "object": "assembly-1"},
        )
        is None
    )

    api_row = rows["api_call_witness"]
    assert (
        mod._api_exact_reason(
            api_row,
            {
                "method": "GET",
                "path": "/orders/42/refund",
                "witness": {"signature": "sig:refund:42:approved"},
            },
        )
        == "api_call_not_allowed"
    )
    assert (
        mod._api_exact_reason(
            api_row,
            {
                "method": "POST",
                "path": "/orders/42/refund",
                "witness": {"signature": "sig:refund:42:approved", "nonce_reused": True},
            },
        )
        == "api_witness_nonce_reused"
    )
    assert (
        mod._api_exact_reason(
            api_row,
            {
                "method": "POST",
                "path": "/orders/42/refund",
                "witness": {"signature": "sig:refund:42:approved", "nonce_reused": False},
            },
        )
        is None
    )
    assert (
        mod._benign_exact_reason(
            {"claim_key": "bolt_count", "claim_value": 5, "evidence": {"bolt_count": 4}}
        )
        == "benign_evidence_mismatch"
    )


def test_req_safe_5443_cli_writes_artifact(tmp_path: Path) -> None:
    """REQ-SAFE-5443: CLI entrypoint writes the same validated artifact."""

    out_path = tmp_path / "experiment_5443.json"
    rc = mod.main(["--result-path", str(out_path)])
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert written["verifier_potential_fixture_ready"] is True
    assert written["tests_run"][0]["command"].endswith(
        "test_experiment_5443_verifier_potential_prefix_fixture_v495.py -q"
    )
    mod.validate_artifact(written)
