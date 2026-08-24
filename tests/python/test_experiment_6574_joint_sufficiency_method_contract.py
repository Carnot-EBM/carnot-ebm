"""Tests for Exp6574 hop-conditioned joint-sufficiency method contract.

Spec refs: REQ-REPORT-6574, REQ-REPORT-6574-GATES,
REQ-REPORT-6574-SOURCES, REQ-REPORT-6574-NODES,
REQ-REPORT-6574-EDGES, REQ-REPORT-6574-REDUCER,
REQ-REPORT-6574-SPLITS-ARMS, REQ-REPORT-6574-FIXTURES,
REQ-REPORT-6574-ATTACKS, REQ-REPORT-6574-ACCEPTANCE,
REQ-REPORT-6574-ATOMIC, SCENARIO-REPORT-6574-NODES,
SCENARIO-REPORT-6574-REDUCER, SCENARIO-REPORT-6574-ABSTAIN,
SCENARIO-REPORT-6574-SPLITS-ARMS, SCENARIO-REPORT-6574-ATTACKS,
SCENARIO-REPORT-6574-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6574_joint_sufficiency_method_contract as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
TESTS_RUN = [{"command": "focused Exp6574 tests", "exit_code": 0}]


def _source_receipts() -> list[dict[str, Any]]:
    return [
        {
            **source,
            "available": True,
            "checked_at_utc": "2026-08-24T00:00:00Z",
            "content_sha256": mod.sha256_text(source["source_id"] + source["title"]),
            "byte_count": len(source["title"]),
            "http_status": 200 if source["source_kind"] == "arxiv_primary" else None,
            "error": "",
        }
        for source in mod.SOURCE_CATALOG
    ]


def _preconditions() -> dict[str, Any]:
    return {
        "run_date": "20260824",
        "planning_date": "20260824",
        "structured_gate": {
            "upstream": "exp6571-v570-evidence-gate-and-retirement-root",
            "path": mod.UPSTREAM_EXP6571_RELATIVE_PATH.as_posix(),
            "artifact_field": "v570_evidence_contract_ready_score",
            "expected": 1.0,
            "observed": 1.0,
            "passed": True,
            "artifact_sha256": "sha256:" + "1" * 64,
        },
        "exp6566_receipt": {
            "path": mod.UPSTREAM_EXP6566_RELATIVE_PATH.as_posix(),
            "sha256": "sha256:" + "6" * 64,
            "source_method_contract_ready_score": 1.0,
        },
        "compiler_and_solver_versions": {
            "compiler": {"name": mod.COMPILER_NAME, "version": mod.COMPILER_VERSION},
            "python": {"version": "3.12.test", "executable": ".venv/bin/python"},
            "z3": {"available": True, "version": "test"},
        },
        "corpus_receipts": [
            {"path": path.as_posix(), "sha256": "sha256:" + str(index) * 64}
            for index, path in enumerate(mod.CORPUS_RELATIVE_PATHS, start=1)
        ],
        "resources": {"cpu": {"count": 8}, "ram": {"total_kib": 1024}, "disk": {"free": 10}},
        "protected_file_hashes": {},
        "no_llm_inference": True,
        "no_hardware_execution": True,
        "hardware_commands_issued": 0,
        "outcome_bearing_extraction_observed": False,
    }


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6574.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        source_review_receipts=_source_receipts(),
        preconditions=_preconditions(),
        run_date="20260824",
    )


def _with_checksum(payload: dict[str, Any]) -> None:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)


def test_req_report_6574_spec_and_required_fields_are_anchored() -> None:
    """REQ-REPORT-6574: the OpenSpec contract exists before implementation."""

    text = (REPO / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6574") :]

    for marker in (
        "REQ-REPORT-6574-GATES",
        "REQ-REPORT-6574-SOURCES",
        "REQ-REPORT-6574-NODES",
        "REQ-REPORT-6574-EDGES",
        "REQ-REPORT-6574-REDUCER",
        "REQ-REPORT-6574-SPLITS-ARMS",
        "REQ-REPORT-6574-FIXTURES",
        "REQ-REPORT-6574-ATTACKS",
        "REQ-REPORT-6574-ACCEPTANCE",
        "REQ-REPORT-6574-ATOMIC",
        "SCENARIO-REPORT-6574-NODES",
        "SCENARIO-REPORT-6574-REDUCER",
        "SCENARIO-REPORT-6574-ABSTAIN",
        "SCENARIO-REPORT-6574-SPLITS-ARMS",
        "SCENARIO-REPORT-6574-ATTACKS",
        "SCENARIO-REPORT-6574-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6574_nodes_bind_source_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6574-NODES: compiler output binds hops to source bytes."""

    artifact = _artifact(tmp_path)
    rows = {
        row["fixture_id"]: row
        for row in artifact["conformance_rows"]
        if row["row_type"] == "conformance_fixture"
    }
    single = rows["valid_single_hop"]
    node = single["nodes"][0]

    assert mod.validate_artifact(artifact) == []
    assert artifact["atomic_obligation_node_schema"]["required_node_fields"] == list(
        mod.REQUIRED_NODE_FIELDS
    )
    assert artifact["atomic_obligation_node_schema"]["compiler_owns_executable_obligation"] is True
    assert (
        artifact["atomic_obligation_node_schema"]["full_constraint_ir_generation_allowed"] is False
    )
    assert node["node_id"] == "valid_single_hop:h0"
    assert node["hop_index"] == 0
    assert node["source_start"] < node["source_end"]
    assert node["source_hash"].startswith("sha256:")
    assert node["source_bytes_hash"].startswith("sha256:")
    assert node["relation"] in mod.WHITELISTED_NODE_RELATIONS
    assert node["compiler_version"] == mod.COMPILER_VERSION
    assert node["executable_obligation_hash"].startswith("sha256:")
    assert node["exact_result"] == "certified_true"
    assert node["counterexample"] is None
    assert node["action"] == "release"

    wrong_span = rows["wrong_span"]["nodes"][0]
    assert wrong_span["action"] == "abstain"
    assert wrong_span["exact_result"] == "source_span_mismatch"
    assert wrong_span["abstention_reason"] == "source_span_mismatch"

    unsupported = rows["unsupported_relation"]["nodes"][0]
    assert unsupported["action"] == "abstain"
    assert unsupported["exact_result"] == "unsupported_relation"

    assert mod._exact_relation_result("less_than", {"left": 3, "right": 5}) == (  # noqa: SLF001
        "certified_true",
        None,
    )
    assert mod._exact_relation_result("equals", {"left": "Ada", "right": "Ben"}) == (  # noqa: SLF001
        "counterexample",
        {"left": "Ada", "right": "Ben"},
    )
    assert mod._exact_relation_result("not_equals", {"left": "Ada", "right": "Ada"}) == (  # noqa: SLF001
        "counterexample",
        {"left": "Ada", "right": "Ada"},
    )
    assert mod._exact_relation_result("contains", {"left": "abc", "right": "b"}) == (  # noqa: SLF001
        "certified_true",
        None,
    )


def test_scenario_report_6574_reducer_releases_only_jointly_sufficient_graphs(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6574-REDUCER/ABSTAIN: unsafe fixture rows abstain."""

    artifact = _artifact(tmp_path)
    fixtures = {
        row["fixture_id"]: row
        for row in artifact["conformance_rows"]
        if row["row_type"] == "conformance_fixture"
    }

    assert set(fixtures) == set(mod.FIXTURE_IDS)
    for fixture_id in ("valid_single_hop", "valid_two_hop", "valid_branched_claim"):
        row = fixtures[fixture_id]
        assert row["action"] == "release"
        assert row["abstention"] is False
        assert row["unsafe_release"] is False
        assert row["reducer_trace"]["required_node_coverage_complete"] is True

    expected_reasons = {
        "missing_hop": "missing_required_hop",
        "wrong_span": "source_span_mismatch",
        "unsupported_relation": "relation_not_whitelisted",
        "contradictory_nodes": "contradictory_nodes",
        "disconnected_graph": "disconnected_required_graph",
        "duplicate_node": "duplicate_node_id",
        "cyclic_dependency": "cyclic_dependency",
    }
    for fixture_id, reason in expected_reasons.items():
        row = fixtures[fixture_id]
        assert row["action"] == "abstain"
        assert row["abstention"] is True
        assert row["unsafe_release"] is False
        assert reason in row["abstention_reasons"]

    optional = mod.build_fixture("missing_hop")
    optional["edges"] = [
        {
            "parent_id": "missing_hop:h0",
            "child_id": "missing_hop:h1",
            "relation_type": "requires_entity_binding",
            "status": "optional",
            "ordering_rule": "parent_hop_before_child",
            "coverage_role": "required_hop_coverage",
            "provenance": "attack fixture",
        }
    ]
    row = mod.evaluate_fixture(optional)
    assert row["action"] == "abstain"
    assert "optional_edge_laundering" in row["abstention_reasons"]


def test_scenario_report_6574_splits_arms_gates_attacks_and_retirement(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6574-SPLITS-ARMS/ATTACKS: commitments are frozen."""

    artifact = _artifact(tmp_path)
    commitment = artifact["frozen_split_and_arm_commitment"]
    gates = artifact["extraction_acceptance_and_retirement_gates"]
    attacks = {
        row["attack_id"]: row for row in artifact["conformance_rows"] if row["row_type"] == "attack"
    }
    retirements = [row for row in artifact["conformance_rows"] if row["row_type"] == "retirement"]

    assert commitment["split_names"] == list(mod.SPLIT_NAMES)
    assert all(row["unit_id"].startswith("sha256:") for row in commitment["split_rows"])
    assert list(commitment["arms"]) == list(mod.ARM_NAMES)
    assert len({arm["matched_commitment_hash"] for arm in commitment["arms"].values()}) == 1
    assert commitment["live_outcomes_observed_before_freeze"] is False
    assert gates["coverage_gate"]["exact_certified_composed_claim_coverage_must_improve"] is True
    assert gates["precision_gate"]["precision_noninferior_required"] is True
    assert gates["safety_gate"]["unsafe_release_must_be_zero"] is True
    assert gates["lineage_and_cost_gate"]["lineage_complete_required"] is True
    assert set(attacks) == set(mod.ATTACK_IDS)
    assert all(row["closed"] for row in attacks.values())
    assert any(row["retire_if_same_verdict"] for row in retirements)

    unequal = deepcopy(artifact)
    unequal["frozen_split_and_arm_commitment"]["arms"]["hop_conditioned_joint"][
        "charged_cost_units"
    ] += 1
    _with_checksum(unequal)
    assert "matched arms diverged" in mod.validate_artifact(unequal)

    open_attack = deepcopy(artifact)
    open_attack["conformance_rows"][-1]["closed"] = False
    _with_checksum(open_attack)
    assert "attack row is not closed" in mod.validate_artifact(open_attack)


def test_req_report_6574_atomic_artifact_validation_and_fail_closed_mutations(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6574-ATOMIC: readiness, checksum, and validators fail closed."""

    result_path = tmp_path / "exp6574.json"
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        source_review_receipts=_source_receipts(),
        preconditions=_preconditions(),
        run_date="20260824",
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert loaded == written
    assert written["status"] == "complete_joint_sufficiency_method_ready"
    assert written["honest_verdict"].startswith("complete_")
    assert written["verdict_class"] is None
    assert written["joint_sufficiency_method_ready_score"] == 1.0
    assert written["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(written)
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert set(written["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(written["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"] is True
    assert written["preconditions_checked"]["no_llm_inference"] is True
    assert written["preconditions_checked"]["no_hardware_execution"] is True
    assert written["gate_check_summary"]["checks_closed"] is True

    classification = adversarial_verify._classify_inference_substrate(written)
    report = adversarial_verify.verify_artifact(result_path)
    assert classification["kind"] == adversarial_verify.SUBSTRATE_KIND_NO_LLM
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert report["flag_count"] == 0

    mutations = [
        (lambda data: data.pop("status"), "missing required fields"),
        (lambda data: data.__setitem__("honest_verdict", "ready"), "terminal prefix"),
        (lambda data: data.__setitem__("verdict_class", "positive"), "closed class"),
        (
            lambda data: data.__setitem__("inference_substrate", "live_llm_inference"),
            "inference_substrate mismatch",
        ),
        (lambda data: data.__setitem__("verifier_is_oracle", False), "must be true"),
        (
            lambda data: data["protected_files_unchanged"].__setitem__("all_unchanged", False),
            "protected files changed",
        ),
        (
            lambda data: data["atomic_obligation_node_schema"]["required_node_fields"].remove(
                "source_bytes_hash"
            ),
            "node schema missing required fields",
        ),
        (
            lambda data: data["dependency_edge_and_joint_reducer_contract"].__setitem__(
                "cycle_handling", "ignore"
            ),
            "cycle handling reopened",
        ),
        (
            lambda data: data["conformance_rows"][3].__setitem__("action", "release"),
            "unsafe fixture released",
        ),
        (
            lambda data: data["gate_check_summary"].__setitem__("failed_checks", ["forced"]),
            "ready score cannot hide failed checks",
        ),
        (
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "joint_sufficiency_method_ready_from_rows", False
            ),
            "aggregate recomputation mismatch",
        ),
        (
            lambda data: data.__setitem__("field_provenance", {}),
            "field_provenance",
        ),
        (
            lambda data: data.__setitem__("field_principles", {}),
            "field_principles",
        ),
        (
            lambda data: data["source_review_receipts"][0].__setitem__("available", False),
            "ready score hides unavailable source",
        ),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(written)
        mutate(candidate)
        _with_checksum(candidate)
        assert any(expected in error for error in mod.validate_artifact(candidate))

    bad_checksum = deepcopy(written)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)


def test_req_report_6574_blocked_gate_and_status_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-6574-GATES: missing upstream or source receipts block readiness."""

    preconditions = _preconditions()
    preconditions["structured_gate"]["passed"] = False
    preconditions["structured_gate"]["observed"] = "missing"
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        source_review_receipts=_source_receipts(),
        preconditions=preconditions,
        run_date="20260824",
    )
    assert blocked["status"] == "blocked_joint_sufficiency_method_missing_prerequisites"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["joint_sufficiency_method_ready_score"] == 0.0
    assert (
        "structured_gate_v570_evidence_contract_ready_score"
        in blocked["gate_check_summary"]["failed_checks"]
    )
    assert mod.validate_artifact(blocked) == []

    missing_source = deepcopy(_source_receipts())
    missing_source[0]["available"] = False
    missing_source[0]["content_sha256"] = "missing"
    blocked_source = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked-source.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        source_review_receipts=missing_source,
        preconditions=_preconditions(),
        run_date="20260824",
    )
    assert blocked_source["verdict_class"] == "blocked"
    assert (
        "source_arxiv:2608.00585_available" in blocked_source["gate_check_summary"]["failed_checks"]
    )

    assert mod._status_and_verdict(True, False, False, []) == (  # noqa: SLF001
        "complete_joint_sufficiency_method_ready",
        "complete_joint_sufficiency_method_ready: source-byte atomic nodes, dependency edges, joint reducer, splits, arms, fixtures, attacks, gates, and retirement rules are frozen",
        None,
    )
    assert mod._status_and_verdict(False, True, False, ["missing"])[2] == "blocked"  # noqa: SLF001
    assert mod._status_and_verdict(False, False, True, ["retro"])[2] == (  # noqa: SLF001
        "disqualified"
    )
    assert mod._status_and_verdict(False, False, False, ["open"])[2] == "partial"  # noqa: SLF001
    summary = mod.gate_check_summary(
        {"source_review_receipts": [], "protected_files_unchanged": {"all_unchanged": False}},
        {"joint_sufficiency_method_ready_from_rows": False, "schema_rows_ready": False},
    )
    assert "protected_files_unchanged" in summary["failed_checks"]
    assert "schema_rows_ready" in summary["failed_checks"]


def test_req_report_6574_defensive_helper_and_validator_branches(tmp_path: Path) -> None:
    """REQ-REPORT-6574-REDUCER/ATOMIC: malformed local contracts fail closed."""

    artifact = _artifact(tmp_path)

    assert mod.sha256_file(None) == "missing"
    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    valid_json = tmp_path / "valid.json"
    valid_json.write_text('{"ok": true}', encoding="utf-8")
    malformed_json = tmp_path / "bad.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert mod._read_json(valid_json) == {"ok": True}  # noqa: SLF001
    assert mod._read_json(malformed_json) == {}  # noqa: SLF001
    assert mod._read_json(tmp_path / "missing-input.json") == {}  # noqa: SLF001
    assert mod._extract_between("a <s> body </s> z", "<s>", "</s>") == "<s> body </s>"  # noqa: SLF001
    assert mod._extract_between("no anchors", "<s>", "</s>") == ""  # noqa: SLF001
    assert mod._exact_relation_result("outside_whitelist", {"left": 1, "right": 1}) == (  # noqa: SLF001
        "unsupported_relation",
        None,
    )
    with pytest.raises(ValueError, match="unknown fixture_id"):
        mod.build_fixture("unknown")

    node = mod.compile_node(mod.build_fixture("valid_single_hop")["nodes"][0])
    malformed = {**node, "action": "defer"}
    edge = {
        "parent_id": node["node_id"],
        "child_id": "missing-node",
        "relation_type": "not_whitelisted",
        "status": "maybe",
        "ordering_rule": "parent_hop_before_child",
        "coverage_role": "required_chain",
        "provenance": "branch test",
    }
    reduction = mod.joint_sufficiency_reduce([malformed], [edge], [0])
    assert reduction["action"] == "abstain"
    assert "node_not_released" in reduction["abstention_reasons"]
    assert "edge_references_unknown_node" in reduction["abstention_reasons"]
    assert "edge_relation_not_whitelisted" in reduction["abstention_reasons"]
    assert "edge_status_invalid" in reduction["abstention_reasons"]

    no_arms = deepcopy(artifact)
    no_arms["frozen_split_and_arm_commitment"]["arms"] = []
    _with_checksum(no_arms)
    assert "matched arms diverged" in mod.validate_artifact(no_arms)
    non_mapping_arm = deepcopy(artifact)
    non_mapping_arm["frozen_split_and_arm_commitment"]["arms"]["no_filter"] = "bad"
    _with_checksum(non_mapping_arm)
    assert "matched arms diverged" in mod.validate_artifact(non_mapping_arm)

    assert mod._status_and_verdict(False, False, False, []) == (  # noqa: SLF001
        "blocked_joint_sufficiency_method_contract",
        "blocked_joint_sufficiency_method_contract: no usable joint-sufficiency rows were available",
        "blocked",
    )

    mutations = [
        (
            lambda data: data["atomic_obligation_node_schema"].__setitem__(
                "compiler_owns_executable_obligation", False
            ),
            "compiler-owned node semantics reopened",
        ),
        (
            lambda data: data["atomic_obligation_node_schema"].__setitem__(
                "full_constraint_ir_generation_allowed", True
            ),
            "full ConstraintIR generation reopened",
        ),
        (
            lambda data: data["dependency_edge_and_joint_reducer_contract"][
                "required_edge_fields"
            ].remove("coverage_role"),
            "edge schema missing required fields",
        ),
        (
            lambda data: data["dependency_edge_and_joint_reducer_contract"].__setitem__(
                "optional_edges_can_satisfy_required_coverage", True
            ),
            "optional edge laundering reopened",
        ),
        (
            lambda data: data["conformance_rows"].pop(0),
            "conformance fixture set mismatch",
        ),
        (
            lambda data: data["conformance_rows"][0].__setitem__("action", "abstain"),
            "safe fixture did not release",
        ),
        (
            lambda data: data["conformance_rows"][4].__setitem__("unsafe_release", True),
            "unsafe fixture released",
        ),
        (
            lambda data: [
                row.__setitem__("attack_id", "wrong")
                for row in data["conformance_rows"]
                if row.get("row_type") == "attack"
            ],
            "attack row set mismatch",
        ),
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(artifact)
        mutate(candidate)
        _with_checksum(candidate)
        assert expected in mod.validate_artifact(candidate)
