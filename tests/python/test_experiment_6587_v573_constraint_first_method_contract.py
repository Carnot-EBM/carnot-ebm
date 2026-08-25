"""Test the V573 constraint-first preregistration without model inference.

Spec refs: REQ-REPORT-6587, REQ-REPORT-6587-PRECONDITIONS,
REQ-REPORT-6587-SOURCES, REQ-REPORT-6587-SOURCE-UNITS,
REQ-REPORT-6587-STAGES, REQ-REPORT-6587-ROUTER,
REQ-REPORT-6587-ARMS, REQ-REPORT-6587-BINDING-AUTHORITY,
REQ-REPORT-6587-METRICS, REQ-REPORT-6587-GATES,
REQ-REPORT-6587-ATTACKS, REQ-REPORT-6587-ATOMIC,
SCENARIO-REPORT-6587-SOURCES, SCENARIO-REPORT-6587-MANIFEST,
SCENARIO-REPORT-6587-STAGES-ROUTER, SCENARIO-REPORT-6587-AUTHORITY,
SCENARIO-REPORT-6587-METRICS-ARMS,
SCENARIO-REPORT-6587-FIXTURES-ATTACKS,
SCENARIO-REPORT-6587-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6574_joint_sufficiency_method_contract as exp6574
from carnot import experiment_6587_v573_constraint_first_method_contract as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
TESTS_RUN = [{"command": "focused Exp6587 fixture", "exit_code": 0, "duration_s": 0.01}]


def _report() -> dict[str, Any]:
    return mod.build_report(REPO, date="20260825", duration_s=1.0, tests_run=TESTS_RUN)


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.artifact_checksum(payload)
    return payload


def test_req_report_6587_spec_declares_method_fields_and_scenarios() -> None:
    """REQ-REPORT-6587: the spec exists before implementation."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6587") :]
    anchors = (
        "REQ-REPORT-6587-PRECONDITIONS",
        "REQ-REPORT-6587-SOURCES",
        "REQ-REPORT-6587-SOURCE-UNITS",
        "REQ-REPORT-6587-STAGES",
        "REQ-REPORT-6587-ROUTER",
        "REQ-REPORT-6587-ARMS",
        "REQ-REPORT-6587-BINDING-AUTHORITY",
        "REQ-REPORT-6587-METRICS",
        "REQ-REPORT-6587-GATES",
        "REQ-REPORT-6587-ATTACKS",
        "REQ-REPORT-6587-ATOMIC",
        "SCENARIO-REPORT-6587-SOURCES",
        "SCENARIO-REPORT-6587-MANIFEST",
        "SCENARIO-REPORT-6587-STAGES-ROUTER",
        "SCENARIO-REPORT-6587-AUTHORITY",
        "SCENARIO-REPORT-6587-METRICS-ARMS",
        "SCENARIO-REPORT-6587-FIXTURES-ATTACKS",
        "SCENARIO-REPORT-6587-ATOMIC",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    )
    for anchor in anchors:
        assert anchor in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6587_sources_are_bounded_and_preconditions_are_pinned(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6587-SOURCES: sources import controls, not claims."""

    report = _report()
    receipts = {row["arxiv_id"]: row for row in report["primary_source_receipts"]}
    excluded = {row["arxiv_id"]: row for row in report["non_imported_claim_rows"]}
    preconditions = report["preconditions_checked"]

    assert set(receipts) == set(mod.REQUIRED_ARXIV_IDS)
    assert set(excluded) == set(mod.REQUIRED_ARXIV_IDS)
    for arxiv_id, receipt in receipts.items():
        assert receipt["stable_url"] == f"https://arxiv.org/abs/{arxiv_id}"
        assert receipt["reference_section_sha256"].startswith("sha256:")
        assert receipt["method_hook"].startswith("Carnot hook:")
        assert receipt["local_cache_status"] in {"cached", "not_cached"}
        assert receipt["receipt_hash"].startswith("sha256:")
        assert excluded[arxiv_id]["claim_imported_into_carnot_evidence"] is False

    assert preconditions["exp6580_receipt"]["sha256"] == mod.sha256_file(
        REPO / mod.EXP6580_RELATIVE_PATH
    )
    assert preconditions["exp6574_receipt"]["sha256"] == mod.sha256_file(
        REPO / mod.EXP6574_RELATIVE_PATH
    )
    assert preconditions["exact_registry"]["registry_sha256"].startswith("sha256:")
    assert preconditions["corpus"]["revision"] == report["source_unit_manifest"]["manifest_hash"]
    assert preconditions["corpus"]["license_spdx"] == "MIT-0"
    assert preconditions["model_inference_invoked"] is False
    assert preconditions["llm_calls_issued"] == 0

    cached = tmp_path / "data" / "paper-2608.05254.txt"
    cached.parent.mkdir()
    cached.write_text("cached source", encoding="utf-8")
    assert mod.local_cache_hits(tmp_path, "2608.05254") == [cached]


def test_scenario_report_6587_manifest_is_exact_balanced_and_outcome_blind() -> None:
    """SCENARIO-REPORT-6587-MANIFEST: 20 frozen units cover all strata."""

    manifest = _report()["source_unit_manifest"]
    units = manifest["units"]

    assert manifest["bounded_unit_count"] == 20
    assert manifest["bounded_unit_count"] >= 16
    assert manifest["selected_without_model_outcomes"] is True
    assert manifest["stratum_counts"] == {"ordinary": 10, "restrictive_cue": 10}
    assert set(manifest["case_class_counts"]) >= {
        "positive_control",
        "unsupported",
        "ambiguous",
        "contradictory",
        "tamper",
    }
    assert {row["split"] for row in units} == {"calibration", "held", "train"}
    for unit in units:
        assert unit["source_bytes_sha256"] == mod.sha256_text(unit["exact_source_bytes"])
        assert unit["task_bytes_sha256"] == mod.sha256_text(unit["exact_task_bytes"])
        assert unit["checker"] == mod.EXACT_CHECKER_NAME
        assert unit["checker_version"] == exp6574.COMPILER_VERSION
        assert unit["selected_without_model_outcome"] is True
        assert unit["model_outcome_fields_accessed"] is False
        assert unit["fixture_hash"].startswith("sha256:")
        assert unit["gold_constraints"]


def test_scenario_report_6587_prompts_router_and_arms_are_frozen() -> None:
    """SCENARIO-REPORT-6587-STAGES-ROUTER: stages and routing stay neutral."""

    report = _report()
    prompts = report["prompt_stage_contract"]
    router = report["router_contract"]
    arms = report["arm_seed_budget_contract"]

    assert set(prompts["prompts"]) == {"direct", "stage1", "stage2"}
    for row in prompts["prompts"].values():
        assert row["sha256"] == mod.sha256_text(row["text"])
        assert row["family_neutral"] is True
    assert prompts["stage1_output_format"] == "plain_text"
    assert prompts["raw_stage_write_before_parse"] is True
    assert prompts["raw_stage1_required"] is True
    assert prompts["raw_stage2_required"] is True
    assert prompts["stage1_answer_requested"] is False
    assert prompts["stage1_answer_transport_allowed"] is False
    assert prompts["constraint_ir_generation_allowed"] is False
    assert prompts["schema_repair_retry_count"] == 0

    assert router["allowed_inputs"] == ["exact_task_bytes", "exact_source_bytes"]
    assert "model_output" in router["forbidden_inputs"]
    assert router["frozen_before_inference"] is True
    routed = mod.route_for_text("Use only the source and determine the result.", "facts")
    direct = mod.route_for_text("Determine the result from the supplied facts.", "facts")
    assert routed["route"] == "cfr"
    assert routed["matched_cues"] == ["only"]
    assert direct == {"route": "direct", "matched_cues": []}

    assert set(arms["arms"]) == {"always_on_cfr", "direct", "routed_cfr"}
    assert arms["stage_token_limits"] == {"direct": 768, "stage1": 256, "stage2": 512}
    assert arms["stage1_tokens_charged"] is True
    assert arms["stage1_latency_charged"] is True
    assert arms["failure_retention_required"] is True
    for key in (
        "unit_order_hash",
        "seed_schedule_hash",
        "decoding_hash",
        "stop_rules_hash",
        "total_token_limit",
        "timeout_s",
    ):
        assert len({row[key] for row in arms["arms"].values()}) == 1


def test_scenario_report_6587_source_binding_and_exact_authority_fail_closed() -> None:
    """SCENARIO-REPORT-6587-AUTHORITY: source and exact checks own release."""

    fixture = exp6574.build_fixture("valid_single_hop")
    node = fixture["nodes"][0]
    proposal = {
        "quoted_span": node["span_text"],
        "source_start": node["source_start"],
        "source_end": node["source_end"],
        "relation": node["relation"],
        "operands": node["operands"],
    }
    supported = mod.bind_constraint_proposal(node["source_text"], proposal)
    assert supported["source_supported"] is True
    assert supported["unsupported"] is False
    assert supported["contradictory"] is False
    assert supported["release_eligible"] is True

    unsupported_proposal = {**proposal, "source_start": node["source_start"] + 1}
    unsupported = mod.bind_constraint_proposal(node["source_text"], unsupported_proposal)
    assert unsupported["source_supported"] is False
    assert unsupported["unsupported"] is True
    assert unsupported["release_eligible"] is False

    bad_fixture = exp6574.build_fixture("contradictory_nodes")
    bad_node = bad_fixture["nodes"][1]
    contradictory = mod.bind_constraint_proposal(
        bad_node["source_text"],
        {
            "quoted_span": bad_node["span_text"],
            "source_start": bad_node["source_start"],
            "source_end": bad_node["source_end"],
            "relation": bad_node["relation"],
            "operands": bad_node["operands"],
        },
    )
    assert contradictory["source_supported"] is True
    assert contradictory["contradictory"] is True
    assert contradictory["action"] == "abstain"

    authority = _report()["source_binding_and_exact_authority_contract"]
    assert authority["model_can_certify_release"] is False
    assert authority["llm_release_authority"] is False
    assert authority["generated_constraint_ir_allowed"] is False
    assert authority["answer_transport_allowed"] is False
    assert authority["external_text_scoring_allowed"] is False
    assert authority["public_arc_solve_scope"] is False


def test_scenario_report_6587_fixtures_metrics_and_gate_rows_close() -> None:
    """SCENARIO-REPORT-6587-METRICS-ARMS: fixtures and reducers are frozen."""

    report = _report()
    fixtures = {row["case_class"]: row for row in report["fixture_replay_rows"]}
    metrics = report["metric_and_acceptance_contract"]
    gates = report["downstream_gate_field_rows"]

    assert fixtures["positive_control"]["observed_action"] == "release"
    for case_class in ("unsupported", "ambiguous", "contradictory", "tamper"):
        assert fixtures[case_class]["observed_action"] == "abstain"
    assert all(row["passed"] is True for row in fixtures.values())
    assert all(row["unsafe_release"] is False for row in fixtures.values())

    assert set(metrics["per_unit_fields"]) >= {
        "abstention",
        "contradictory_constraint_count",
        "exact_success",
        "failure",
        "latency_s",
        "stage1_precision",
        "stage1_recall",
        "total_tokens",
        "unsafe_release",
        "unsupported_constraint_count",
    }
    assert metrics["paired_unit_key"] == "unit_id"
    assert metrics["paired_uncertainty"]["method"] == "paired_unit_bootstrap_ci95"
    assert metrics["later_positive_verdict_class"] == "circular_positive"
    assert metrics["ready_contract_verdict_class"] is None
    assert metrics["acceptance_thresholds"]["unsafe_release_increase_max"] == 0
    assert metrics["retirement_rules"]

    method_gate = next(row for row in gates if row["artifact_field"] == mod.READY_FIELD)
    assert method_gate["owner_task_id"] == "exp6587-v573-constraint-first-method-contract"
    assert set(method_gate["consumer_task_ids"]) == {
        "exp6588-qwen36-constraint-first-stream",
        "exp6589-gemma4-31b-constraint-first-stream",
        "exp6590-independent-constraint-first-comparison",
    }
    assert all(row["owner_declared"] is True for row in gates)
    assert all(row["all_consumers_declared"] is True for row in gates)


def test_scenario_report_6587_attacks_make_readiness_fail_closed() -> None:
    """SCENARIO-REPORT-6587-FIXTURES-ATTACKS: every required attack blocks."""

    report = _report()
    attacks = {row["attack_id"]: row for row in report["attack_rows"]}
    assert set(attacks) == set(mod.REQUIRED_ATTACK_IDS)
    assert all(row["passed"] is True for row in attacks.values())
    assert all(row["candidate_ready_score"] == 0.0 for row in attacks.values())
    assert mod.readiness_reducer(report)["ready_score"] == 1.0

    mutations = []
    candidate = deepcopy(report)
    candidate["source_unit_manifest"]["selected_without_model_outcomes"] = False
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["prompt_stage_contract"]["prompts"]["direct"]["family_neutral"] = False
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["prompt_stage_contract"]["stage1_answer_requested"] = True
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["prompt_stage_contract"]["constraint_ir_generation_allowed"] = True
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["prompt_stage_contract"]["raw_stage1_required"] = False
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["router_contract"]["allowed_inputs"].append("model_output")
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["source_binding_and_exact_authority_contract"]["llm_release_authority"] = True
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["arm_seed_budget_contract"]["stage1_tokens_charged"] = False
    mutations.append(candidate)
    candidate = deepcopy(report)
    candidate["downstream_gate_field_rows"][0]["artifact_field"] = (
        "v573_constraint_first_method_ready_scor"
    )
    mutations.append(candidate)

    assert len(mutations) == len(mod.REQUIRED_ATTACK_IDS)
    assert all(mod.readiness_reducer(candidate)["ready_score"] == 0.0 for candidate in mutations)


def test_scenario_report_6587_atomic_null_artifact_validates(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6587-ATOMIC: one null no-LLM artifact recomputes."""

    report = _report()
    output = tmp_path / "experiment_6587.json"
    receipt = mod.atomic_write_report(output, report)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == report
    assert receipt["atomic_replace"] is True
    assert receipt["file_fsync"] is True
    assert receipt["directory_fsync"] is True
    assert receipt["output_sha256"] == mod.sha256_file(output)
    assert report["status"] == "complete_v573_constraint_first_method_ready"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] is None
    assert report[mod.READY_FIELD] == 1.0
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is True
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["reproducibility_checksum"] == mod.artifact_checksum(report)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert mod.validate_report(report) == []

    classification = adversarial_verify._classify_inference_substrate(report)
    verification = adversarial_verify.verify_artifact(output)
    assert classification["kind"] == adversarial_verify.SUBSTRATE_KIND_NO_LLM
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert verification["flag_count"] == 0


def test_req_report_6587_validator_rejects_tamper_and_bad_terminal_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6587-ATOMIC: checksum and terminal fields fail closed."""

    report = _report()
    cases = []
    candidate = deepcopy(report)
    del candidate["status"]
    cases.append((candidate, "missing required fields: status"))
    cases.append(
        (
            _rehash({**deepcopy(report), "inference_substrate": "wrong"}),
            "inference_substrate mismatch",
        )
    )
    cases.append(
        (
            _rehash({**deepcopy(report), "verifier_is_oracle": False}),
            "verifier_is_oracle must be true",
        )
    )
    cases.append((_rehash({**deepcopy(report), "duration_s": 0.0}), "duration_s must be positive"))
    cases.append(
        (
            _rehash({**deepcopy(report), mod.READY_FIELD: 0.0}),
            f"{mod.READY_FIELD} mismatch",
        )
    )
    cases.append(
        (
            _rehash({**deepcopy(report), "verdict_class": "positive"}),
            "ready contract verdict_class must be null",
        )
    )
    candidate = deepcopy(report)
    candidate["protected_files_unchanged"]["all_unchanged"] = False
    cases.append((_rehash(candidate), "protected_files_unchanged failed"))
    candidate = deepcopy(report)
    candidate["field_provenance"].pop("status")
    cases.append((_rehash(candidate), "field_provenance missing required fields"))
    cases.append(
        (
            {**deepcopy(report), "reproducibility_checksum": "sha256:stale"},
            "reproducibility_checksum mismatch",
        )
    )

    for candidate, expected in cases:
        assert expected in mod.validate_report(candidate)

    bad = _rehash({**deepcopy(report), "duration_s": 0.0})
    with pytest.raises(ValueError, match="duration_s must be positive"):
        mod.atomic_write_report(REPO / "results" / "should-not-write.json", bad)

    assert mod._extract_reference_section("missing V573 anchors") == ""  # noqa: SLF001
    failed_attacks = mod.build_attack_rows()
    failed_attacks[0]["passed"] = False
    monkeypatch.setattr(mod, "build_attack_rows", lambda: failed_attacks)
    blocked = mod.build_report(REPO, date="20260825", duration_s=1.0, tests_run=TESTS_RUN)
    assert blocked["status"] == "blocked_v573_constraint_first_method_contract"
    assert blocked["honest_verdict"].startswith("blocked_v573_constraint_first_method_contract:")
    assert blocked["verdict_class"] == "blocked"
    assert blocked[mod.READY_FIELD] == 0.0
