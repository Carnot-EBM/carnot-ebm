"""Tests for Exp6566 proof-obligation and graph-Potts method contract.

Spec refs: REQ-REPORT-6566, SCENARIO-REPORT-6566-SOURCES,
SCENARIO-REPORT-6566-PROOF, SCENARIO-REPORT-6566-SPLITS-GRAPH,
SCENARIO-REPORT-6566-POTTS, SCENARIO-REPORT-6566-MATCHED-DOSE,
SCENARIO-REPORT-6566-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6566_proof_obligation_and_graph_potts_method_contract as mod
from scripts import adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6566", "exit_code": 0}]


def _source_receipts() -> list[dict[str, Any]]:
    return [
        {
            **source,
            "url": source["arxiv_url"],
            "available": True,
            "checked_at_utc": "2026-08-23T00:00:00Z",
            "content_sha256": mod.sha256_text(source["arxiv_id"] + source["title"]),
            "byte_count": len(source["title"]),
        }
        for source in mod.SOURCE_CATALOG
    ]


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "exp6566.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        source_review_receipts=_source_receipts(),
        run_date="20260823",
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def test_req_report_6566_spec_declares_required_contract() -> None:
    """REQ-REPORT-6566: OpenSpec owns the V569 method contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6566") : text.index("REQ-REPORT-6565")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-6566-SOURCES",
        "SCENARIO-REPORT-6566-PROOF",
        "SCENARIO-REPORT-6566-SPLITS-GRAPH",
        "SCENARIO-REPORT-6566-POTTS",
        "SCENARIO-REPORT-6566-MATCHED-DOSE",
        "SCENARIO-REPORT-6566-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6566_sources_and_preconditions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6566-SOURCES: source and tool receipts gate readiness."""

    artifact = _artifact(tmp_path)
    assert mod.validate_artifact(artifact) == []
    assert mod.sha256_bytes(b"exp6566").startswith("sha256:")
    assert mod.sha256_file(None) == "missing"
    assert mod.sha256_file(Path("/tmp/definitely-missing-exp6566-file")) == "missing"

    assert [row["arxiv_id"] for row in artifact["source_review_receipts"]] == [
        source["arxiv_id"] for source in mod.SOURCE_CATALOG
    ]
    assert all(row["available"] for row in artifact["source_review_receipts"])
    assert all(
        row["content_sha256"].startswith("sha256:") for row in artifact["source_review_receipts"]
    )
    assert artifact["preconditions_checked"]["no_llm_inference"] is True
    assert artifact["preconditions_checked"]["hardware_commands_issued"] == 0
    assert artifact["preconditions_checked"]["compiler"]["name"] == mod.COMPILER_NAME
    assert artifact["preconditions_checked"]["corpus"]["drift_fixture"]["path"].endswith(
        "v566_drift_bench_external_slice.jsonl"
    )

    blocked_sources = deepcopy(_source_receipts())
    blocked_sources[0]["available"] = False
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        source_review_receipts=blocked_sources,
        run_date="20260823",
    )
    assert blocked["status"] == "blocked_source_method_contract_missing_prerequisites"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["source_method_contract_ready_score"] == 0.0
    assert "source_2608.17941_available" in blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_6566_proof_obligation_contract(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6566-PROOF: claims carry source spans and compiler obligations."""

    artifact = _artifact(tmp_path)
    contract = artifact["proof_obligation_schema_and_compiler_contract"]
    rows = {
        row["unit_id"]: row for row in artifact["conformance_rows"] if row["row_type"] == "proof"
    }

    assert contract["full_constraint_ir_generation_allowed"] is False
    assert contract["schema_validity_is_semantic_validity"] is False
    assert contract["compiler_owns_executable_obligation"] is True
    assert set(contract["required_claim_fields"]) >= {
        "source_start",
        "source_end",
        "source_sha256",
        "typed_variables",
        "relation",
        "compiler_version",
        "executable_obligation_hash",
        "exact_result",
        "release_action",
    }

    supported = rows["proof-age-01"]
    assert supported["source_start"] < supported["source_end"]
    assert supported["source_sha256"].startswith("sha256:")
    assert supported["relation"] in mod.WHITELISTED_RELATIONS
    assert supported["executable_obligation_hash"].startswith("sha256:")
    assert supported["exact_result"] == "certified_true"
    assert supported["release_action"] == "release"
    assert supported["abstention"] is False
    assert supported["full_constraint_ir"] is None

    counterexample = rows["proof-age-02"]
    assert counterexample["exact_result"] == "counterexample"
    assert counterexample["release_action"] == "reject"
    assert counterexample["counterexample"]["left"] == 9

    abstained = rows["proof-age-03"]
    assert abstained["abstention"] is True
    assert abstained["release_action"] == "abstain"
    assert abstained["abstention_reason"] == "relation_not_whitelisted"

    with pytest.raises(ValueError, match="source span text mismatch"):
        mod.compile_claim(
            {
                "unit_id": "bad-span",
                "source_text": "Ada is older than Ben.",
                "source_start": 0,
                "source_end": 3,
                "span_text": "Ben",
                "typed_variables": {"left": "person", "right": "person"},
                "relation": "greater_than",
                "operands": {"left": 7, "right": 5},
            }
        )
    assert mod._exact_relation_result("less_than", {"left": 3, "right": 5}) == (  # noqa: SLF001
        "certified_true",
        None,
    )
    assert mod._exact_relation_result("equals", {"left": 3, "right": 4}) == (  # noqa: SLF001
        "counterexample",
        {"left": 3, "right": 4},
    )
    assert mod._exact_relation_result("not_equals", {"left": 3, "right": 3}) == (  # noqa: SLF001
        "counterexample",
        {"left": 3, "right": 3},
    )
    assert mod._exact_relation_result("subset_of", {"left": "A", "right": "B"}) == (  # noqa: SLF001
        "certified_true",
        None,
    )


def test_scenario_report_6566_split_graph_and_leakage_contract(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6566-SPLITS-GRAPH: graph features exclude leakage."""

    artifact = _artifact(tmp_path)
    split = artifact["frozen_split_and_unit_commitment"]
    graph = artifact["graph_feature_and_leakage_contract"]
    rows = [row for row in artifact["conformance_rows"] if row["row_type"] == "graph"]

    assert split["slice_names"] == list(mod.SLICE_NAMES)
    assert split["family_blind"] is True
    assert all(row["unit_id"].startswith("sha256:") for row in split["unit_rows"])
    assert graph["allowed_features"] == list(mod.ALLOWED_GRAPH_FEATURES)
    assert graph["forbidden_features"] == list(mod.FORBIDDEN_GRAPH_FEATURES)
    assert graph["forbidden_features_present"] == []
    assert graph["connected_component_count"] == 1
    assert len(rows) >= 2
    assert all("model_identity" not in row["feature_keys"] for row in rows)
    assert all("target_label" not in row["feature_keys"] for row in rows)

    disconnected = deepcopy(artifact)
    disconnected["graph_feature_and_leakage_contract"]["connected_component_count"] = 2
    _with_checksum(disconnected)
    assert "graph is disconnected" in mod.validate_artifact(disconnected)


def test_scenario_report_6566_potts_beta_binomial_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6566-POTTS: estimator equations and rows recompute."""

    artifact = _artifact(tmp_path)
    equations = artifact["potts_beta_binomial_equations"]
    row = next(row for row in artifact["conformance_rows"] if row["row_type"] == "potts")

    assert (
        equations["potts_prior"] == "P(z) proportional to exp(beta * sum_edges w_ij * 1[z_i=z_j])"
    )
    assert equations["beta_binomial_mean"] == "(alpha_s + success_s) / (alpha_s + beta_s + n_s)"
    assert equations["online_mean_field_update"].startswith("q_i")
    assert equations["clamps"] == mod.NUMERICAL_CLAMPS
    assert (
        equations["restart_state"]["state_hash"] == equations["rollback_state"]["pre_update_hash"]
    )
    assert row["converged"] is True
    assert row["iterations"] <= equations["iteration_cap"]
    assert row["posterior_means_by_state"] == {"easy": 0.75, "hard": 0.25}
    assert row["state_probabilities"]["u1"]["easy"] == pytest.approx(0.8175744761936437)
    assert row["state_probabilities"]["u1"]["hard"] == pytest.approx(0.18242552380635635)

    nonconverged = deepcopy(artifact)
    for candidate in nonconverged["conformance_rows"]:
        if candidate["row_type"] == "potts":
            candidate["converged"] = False
    _with_checksum(nonconverged)
    assert "potts mean-field row did not converge" in mod.validate_artifact(nonconverged)


def test_scenario_report_6566_matched_dose_gates_and_attacks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6566-MATCHED-DOSE: arms, gates, attacks, and retirements freeze."""

    artifact = _artifact(tmp_path)
    arms = artifact["matched_dose_arm_contract"]["arms"]
    gates = artifact["extraction_and_csl_acceptance_gates"]
    attacks = {
        row["attack_id"]: row for row in artifact["conformance_rows"] if row["row_type"] == "attack"
    }
    retirements = [row for row in artifact["conformance_rows"] if row["row_type"] == "retirement"]

    assert list(arms) == list(mod.ARM_NAMES)
    doses = {tuple(sorted(arms[name]["dose"].items())) for name in arms}
    assert len(doses) == 1
    assert gates["extraction"]["zero_unsafe_release_required"] is True
    assert gates["csl"]["future_support_required"] is True
    assert gates["csl"]["noninferior_charged_cost_required"] is True
    assert set(attacks) == set(mod.ATTACK_IDS)
    assert all(row["closed"] for row in attacks.values())
    assert any(row["retire_if_same_verdict"] for row in retirements)

    unequal = deepcopy(artifact)
    unequal["matched_dose_arm_contract"]["arms"]["graph_potts"]["dose"]["charged_dose"] += 1
    _with_checksum(unequal)
    assert "matched-dose arms are unequal" in mod.validate_artifact(unequal)


def test_scenario_report_6566_atomic_artifact_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6566-ATOMIC: atomic artifact and validators fail closed."""

    result_path = tmp_path / "exp6566.json"
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        source_review_receipts=_source_receipts(),
        run_date="20260823",
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert loaded["reproducibility_checksum"] == written["reproducibility_checksum"]
    assert written["status"] == "complete_source_method_contract_ready"
    assert written["honest_verdict"].startswith("complete_")
    assert written["verdict_class"] is None
    assert written["source_method_contract_ready_score"] == 1.0
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"] is True
    assert written["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(written)
    assert set(written["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(written["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    classification = av._classify_inference_substrate(written)
    report = av.verify_artifact(result_path)
    assert classification["kind"] == av.SUBSTRATE_KIND_NO_LLM
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in {flag["kind"] for flag in report["flags"]}

    mutations = [
        (lambda data: data.pop("status"), "missing required fields"),
        (
            lambda data: data.__setitem__("honest_verdict", "ready"),
            "honest_verdict lacks terminal prefix",
        ),
        (
            lambda data: data.__setitem__("verdict_class", "positive"),
            "verdict_class is outside closed class",
        ),
        (
            lambda data: data.__setitem__("inference_substrate", "live_llm_inference"),
            "inference_substrate mismatch",
        ),
        (
            lambda data: data.__setitem__("verifier_is_oracle", False),
            "verifier_is_oracle must be true",
        ),
        (
            lambda data: data["protected_files_unchanged"].__setitem__("all_unchanged", False),
            "protected files changed",
        ),
        (
            lambda data: data["proof_obligation_schema_and_compiler_contract"].__setitem__(
                "full_constraint_ir_generation_allowed", True
            ),
            "full ConstraintIR generation reopened",
        ),
        (
            lambda data: data["proof_obligation_schema_and_compiler_contract"].__setitem__(
                "compiler_owns_executable_obligation", False
            ),
            "compiler-owned obligation boundary opened",
        ),
        (
            lambda data: data["proof_obligation_schema_and_compiler_contract"].__setitem__(
                "schema_validity_is_semantic_validity", True
            ),
            "schema-valid semantic-invalid boundary opened",
        ),
        (
            lambda data: data["graph_feature_and_leakage_contract"].__setitem__(
                "forbidden_features_present", ["model_identity"]
            ),
            "graph leakage features present",
        ),
        (
            lambda data: next(
                row for row in data["conformance_rows"] if row["row_type"] == "attack"
            ).__setitem__("closed", False),
            "attack row is not closed",
        ),
        (
            lambda data: data["gate_check_summary"].__setitem__("failed_checks", ["forced"]),
            "ready score cannot be open with failed checks",
        ),
        (
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "source_method_contract_ready_from_rows", False
            ),
            "ready score must derive from aggregate recomputation",
        ),
        (
            lambda data: data.__setitem__("field_provenance", {}),
            "field_provenance must cover required fields",
        ),
        (
            lambda data: data.__setitem__("field_principles", {}),
            "field_principles must cover required fields",
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

    assert mod._status_and_verdict(True, False, []) == (  # noqa: SLF001
        "complete_source_method_contract_ready",
        "complete_source_method_contract_ready: proof-obligation schema, immutable splits, graph features, Potts equations, matched-dose arms, gates, attacks, and retirement rules are frozen",
        None,
    )
    assert mod._status_and_verdict(False, True, ["missing"]) == (  # noqa: SLF001
        "blocked_source_method_contract_missing_prerequisites",
        "blocked_source_method_contract_missing_prerequisites: required source, tool, corpus, equation, fixture, or field is missing",
        "blocked",
    )
    assert mod._status_and_verdict(False, False, ["gate"]) == (  # noqa: SLF001
        "partial_source_method_contract",
        "partial_source_method_contract: usable preregistration exists but one or more source, equation, fixture, attack, or field checks failed",
        "partial",
    )
    assert mod._status_and_verdict(False, False, []) == (  # noqa: SLF001
        "blocked_source_method_contract",
        "blocked_source_method_contract: no usable method contract rows were available",
        "blocked",
    )
    summary = mod.gate_check_summary(
        {"source_review_receipts": [], "protected_files_unchanged": {"all_unchanged": False}},
        {"source_method_contract_ready_from_rows": False, "graph_ready": False},
    )
    assert "protected_files_unchanged" in summary["failed_checks"]
    assert "graph_ready" in summary["failed_checks"]
