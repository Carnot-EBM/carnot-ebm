"""Tests for the immutable V570 evidence, gate, failure, and retirement root."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6571_v570_evidence_gate_and_retirement_root as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
TESTS_RUN = [{"command": "focused Exp6571 tests", "exit_code": 0}]


def _fake_live_checks() -> dict[str, dict[str, Any]]:
    """Return deterministic checker receipts with the real missing-path shape."""

    rows: dict[str, dict[str, Any]] = {}
    for artifact in mod.V569_ARTIFACTS:
        missing = artifact.exp_id == "exp6569"
        skipped = artifact.exp_id in {"exp6568", "exp6570"}
        rows[artifact.exp_id] = {
            "adversarial": {
                "command": f"adversarial {artifact.exp_id}",
                "exit_code": 0,
                "duration_s": 0.01,
                "loaded": not missing,
                "flags": [],
            },
            "row_consistency": {
                "command": f"row-lint {artifact.exp_id}",
                "exit_code": 1 if missing else 0,
                "duration_s": 0.01,
                "status": "unreadable" if missing else ("skipped" if skipped else "ok"),
                "findings": [],
            },
            "artifact_convention": {
                "command": f"artifact-convention {artifact.exp_id}",
                "exit_code": 0,
                "duration_s": 0.001,
                "status": "EXPECTED_MISSING" if missing else "CHECKABLE",
                "reason": "expected artifact is absent"
                if missing
                else "rows or blocker diagnostics exist",
            },
        }
    return rows


def _fake_preconditions() -> dict[str, Any]:
    """Return a small no-LLM, no-hardware precondition receipt for unit tests."""

    return {
        "git_status": {"status_short": ""},
        "resources": {"cpu": {}, "ram": {}, "disk": {}, "python": {}},
        "rust": {},
        "pyo3": {},
        "z3": {},
        "model_cache_paths": [],
        "artifact_path_and_hash_receipts": [],
        "protected_file_hashes_before": {},
        "monotonic_timer_resolution_s": 1e-9,
        "architecture_freshness": {"architecture_checked": True},
        "network_status": {"checked": True},
        "llm_load_performed": False,
        "hardware_command_performed": False,
    }


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    """Build the real-input artifact without writing tracked state."""

    return mod.build_artifact(
        repo_root=REPO,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_live_checks(),
        preconditions=_fake_preconditions(),
        run_date="20260824",
    )


def _with_checksum(payload: dict[str, Any]) -> None:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)


def test_req_report_6571_spec_and_required_principles_are_anchored() -> None:
    """REQ-REPORT-6571: the OpenSpec contract exists before implementation."""

    text = (REPO / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6571") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-6571-IMPORT",
        "REQ-REPORT-6571-LIVE",
        "REQ-REPORT-6571-GGUF",
        "REQ-REPORT-6571-VERDICT",
        "REQ-REPORT-6571-GATES",
        "REQ-REPORT-6571-FAILURES",
        "REQ-REPORT-6571-BOUNDARIES",
        "REQ-REPORT-6571-RUST",
        "REQ-REPORT-6571-ATTACKS",
        "REQ-REPORT-6571-ATOMIC",
        "SCENARIO-REPORT-6571-IMPORT",
        "SCENARIO-REPORT-6571-LIVE",
        "SCENARIO-REPORT-6571-GGUF",
        "SCENARIO-REPORT-6571-GATES",
        "SCENARIO-REPORT-6571-BOUNDARIES",
        "SCENARIO-REPORT-6571-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section or field in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6571_import_and_live_rows_classify_v569_honestly(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6571-IMPORT/LIVE: exact imports preserve every outcome."""

    assert mod.validate_artifact(artifact) == []
    rows = {row["exp_id"]: row for row in artifact["v569_artifact_eligibility_rows"]}
    live = {row["exp_id"]: row for row in artifact["live_verifier_and_duration_rows"]}

    assert list(rows) == [item.exp_id for item in mod.V569_ARTIFACTS]
    assert rows["exp6565"]["disposition"] == "usable_v569_evidence_contract"
    assert rows["exp6566"]["disposition"] == "usable_v569_method_contract"
    assert rows["exp6567"]["disposition"] == "blocked_hash_only_gguf_admission"
    assert rows["exp6568"]["source_verdict_class"] is None
    assert rows["exp6568"]["verdict_class"] == "blocked"
    assert rows["exp6568"]["disposition"] == "corrected_blocked_gate_import"
    assert rows["exp6569"]["expected_path"] == (
        "results/experiment_6569_source_span_proof_obligation_extractor.json"
    )
    assert rows["exp6569"]["exists"] is False
    assert rows["exp6569"]["sha256"] == "missing"
    assert rows["exp6569"]["verdict_class"] == "blocked"
    assert rows["exp6569"]["disposition"] == "missing_not_null"
    assert rows["exp6570"]["disposition"] == "valid_blocked_independent_audit"
    assert all(row["eligible_for_v570_contract"] for row in rows.values())

    assert rows["exp6565"]["stamped_live_flag_disagreement"] is True
    assert {flag["kind"] for flag in rows["exp6565"]["stamped_flags"]} >= {"DURATION_TOO_SHORT"}
    assert live["exp6569"]["artifact_loaded_by_adversarial_verifier"] is False
    assert live["exp6569"]["row_consistency_status"] == "unreadable"
    assert live["exp6569"]["artifact_convention_status"] == "EXPECTED_MISSING"
    for row in live.values():
        assert row["live_verifier_command"]
        assert row["row_consistency_command"]
        assert row["artifact_convention_command"]
        assert row["live_verifier_duration_s"] >= 0.0
        assert row["row_consistency_duration_s"] >= 0.0
        assert row["artifact_convention_duration_s"] >= 0.0


def test_scenario_report_6571_gguf_root_cause_is_derived_from_rows(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6571-GGUF: the observed path defect controls the retry."""

    root = artifact["gguf_admission_root_cause"]

    assert root["source_exp_id"] == "exp6567"
    assert root["source_artifact_sha256"].startswith("sha256:")
    assert root["resolved_blob_count"] == 3
    assert root["all_large_blobs_resolved"] is True
    assert root["all_embedded_tokenizers_passed"] is True
    assert root["cuda_runtime_passed"] is True
    assert root["sequential_memory_conditions_passed"] is True
    assert root["failed_precondition"] == "model_identity_and_file_shape"
    assert root["all_language_model_file_false"] is True
    assert root["all_quantization_known_false"] is True
    assert root["all_paths_hash_only"] is True
    assert root["generation_row_count"] == 0
    assert root["generation_ran"] is False
    assert len(root["per_model_rows"]) == 3
    assert all(row["language_model_file"] is False for row in root["per_model_rows"])
    assert all(row["quantization_known"] is False for row in root["per_model_rows"])


def test_scenario_report_6571_gates_failures_boundaries_and_attacks_close(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6571-GATES/BOUNDARIES: exact changed mechanisms are frozen."""

    gates = artifact["v570_gate_contract_rows"]
    assert len(gates) == 4
    assert all(row["upstream_in_active_roadmap"] for row in gates)
    assert all(row["artifact_field_declared_by_upstream"] for row in gates)
    assert all(row["retired_upstream"] is False for row in gates)
    assert all(row["exact_field_spelling"] for row in gates)

    priors = artifact["prior_failure_and_retirement_rows"]
    assert len(priors) == 5
    assert all(row["complete_prior_failure_contract"] for row in priors)
    assert all(row["changed_mechanism"] for row in priors)
    assert all(row["mechanical_repeat_retirement_rule"] for row in priors)
    assert all(row["retired_dependency_chain"] is False for row in priors)

    boundary = artifact["model_arc_and_hardware_boundary"]
    assert len(boundary["v570_task_manifest"]) == 4
    assert boundary["MODEL_SPECS"] == list(mod.MANDATED_MODEL_IDS)
    assert boundary["model_rule"]["content_derived_gguf_metadata_required"] is True
    assert boundary["model_rule"]["actual_execution_required"] is True
    assert boundary["legacy_model_policy"]["legacy_substitution_allowed"] is False
    assert boundary["extraction_boundary"]["joint_sufficiency_required"] is True
    assert boundary["extraction_boundary"]["retired_constraintir_reuse_allowed"] is False
    assert boundary["arc_boundary"]["solve_claim_allowed"] is False
    assert boundary["hardware_boundary"]["unchanged_hardware_command_allowed"] is False
    assert boundary["hardware_boundary"]["exp6571_hardware_command_count"] == 0
    assert boundary["rust_fusion_boundary"]["materially_different_from_exp6563_6564"] is True
    assert boundary["rust_fusion_boundary"]["retire_on_repeat_no_benefit_or_nfr01_miss"] is True

    assert len(artifact["attack_rows"]) == len(mod.ATTACK_NAMES)
    assert all(row["passed"] for row in artifact["attack_rows"])
    assert artifact["v570_evidence_contract_ready_score"] == 1.0
    assert artifact["rust_fusion_reopen_ready_score"] == 1.0
    assert artifact["verdict_class"] is None
    assert artifact["honest_verdict"].startswith("complete_")


def test_scenario_report_6571_atomic_output_and_attack_validation(
    artifact: dict[str, Any], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-6571-ATOMIC: checksum, provenance, and attacks fail closed."""

    result_path = tmp_path / "exp6571.json"
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_live_checks(),
        preconditions=_fake_preconditions(),
        run_date="20260824",
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert loaded == written
    assert written["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(written)
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert set(written["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(written["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"] is True
    assert written["preconditions_checked"]["llm_load_performed"] is False
    assert written["preconditions_checked"]["hardware_command_performed"] is False

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
            lambda data: data["v569_artifact_eligibility_rows"][0].__setitem__(
                "sha256", "sha256:alias"
            ),
            "hash alias",
        ),
        (
            lambda data: data["v569_artifact_eligibility_rows"][4].__setitem__(
                "disposition", "clean_null"
            ),
            "Exp6569 must remain missing",
        ),
        (
            lambda data: data["v569_artifact_eligibility_rows"][3].__setitem__(
                "verdict_class", None
            ),
            "Exp6568 corrected class",
        ),
        (
            lambda data: data["v570_gate_contract_rows"][0].__setitem__(
                "exact_field_spelling", False
            ),
            "gate field drift",
        ),
        (
            lambda data: data["prior_failure_and_retirement_rows"][0].__setitem__(
                "changed_mechanism", False
            ),
            "changed mechanism",
        ),
        (
            lambda data: data["gguf_admission_root_cause"].__setitem__("generation_ran", True),
            "false model admission",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"]["MODEL_SPECS"].append(
                "legacy/model"
            ),
            "model identities changed",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"]["arc_boundary"].__setitem__(
                "solve_claim_allowed", True
            ),
            "ARC solve boundary",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"]["hardware_boundary"].__setitem__(
                "exp6571_hardware_command_count", 1
            ),
            "hardware command boundary",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"][
                "rust_fusion_boundary"
            ].__setitem__("materially_different_from_exp6563_6564", False),
            "Rust fusion boundary",
        ),
        (
            lambda data: data["protected_files_unchanged"].__setitem__("all_unchanged", False),
            "protected files changed",
        ),
        (
            lambda data: data["attack_rows"][0].__setitem__("passed", False),
            "attack matrix",
        ),
        (
            lambda data: data.__setitem__("field_provenance", {}),
            "field_provenance",
        ),
        (
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "v570_evidence_contract_ready_from_rows", False
            ),
            "aggregate recomputation mismatch",
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


def test_req_report_6571_unexpected_missing_input_blocks_but_exp6569_does_not(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6571-IMPORT/VERDICT: only unexpected prerequisites block."""

    paths = mod.default_v569_paths(REPO)
    paths["exp6565"] = tmp_path / "missing-exp6565.json"
    blocked = mod.build_artifact(
        repo_root=REPO,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_live_checks(),
        preconditions=_fake_preconditions(),
        artifact_paths=paths,
        run_date="20260824",
    )

    assert blocked["status"] == "blocked_v570_evidence_contract_missing_prerequisites"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["v570_evidence_contract_ready_score"] == 0.0
    assert blocked["rust_fusion_reopen_ready_score"] == 1.0
    assert (
        "exp6565_expected_artifact_classification" in blocked["gate_check_summary"]["failed_checks"]
    )
    assert mod.validate_artifact(blocked) == []


def test_req_report_6571_helper_edges_fail_closed(artifact: dict[str, Any]) -> None:
    """REQ-REPORT-6571-ATTACKS: malformed shapes cannot become ready."""

    assert mod._closed_verdict_class(None) is None
    assert mod._closed_verdict_class("blocked") == "blocked"
    assert mod._closed_verdict_class("positive") == "disqualified"
    assert mod._is_hash_only_path("/cache/blobs/" + "a" * 64) is True
    assert mod._is_hash_only_path("/cache/model.gguf") is False
    assert mod._status_and_verdict(True, False, False)[2] is None
    assert mod._status_and_verdict(False, True, False)[2] == "disqualified"
    assert mod._status_and_verdict(False, False, True)[2] == "blocked"
    assert mod._status_and_verdict(False, False, False)[2] == "partial"

    no_rows = deepcopy(artifact)
    no_rows["per_unit_rows"] = []
    _with_checksum(no_rows)
    assert "aggregate-only evidence root" in mod.validate_artifact(no_rows)

    malformed = deepcopy(artifact)
    malformed["v570_gate_contract_rows"] = ["not-a-row"]
    malformed["prior_failure_and_retirement_rows"] = ["not-a-row"]
    _with_checksum(malformed)
    errors = mod.validate_artifact(malformed)
    assert "gate contract row must be a mapping" in errors
    assert "prior failure row must be a mapping" in errors


def test_req_report_6571_helper_and_validator_branches_are_covered(
    artifact: dict[str, Any], tmp_path: Path
) -> None:
    """REQ-REPORT-6571-ATOMIC/ATTACKS: defensive helper branches stay executable."""

    assert mod.sha256_bytes(b"exp6571").startswith("sha256:")
    assert mod.sha256_file(None) == "missing"
    assert mod._command_text(["python", "two words"]) == "python 'two words'"
    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert mod._read_json(malformed_json) == {}
    assert mod.load_json(malformed_json) == {}

    assert mod._contains_rows({}, depth=5) is False
    assert mod._contains_rows([{"nested_rows": [{"x": 1}]}]) is True
    assert mod._contains_rows({"outer": {"nested_rows": [{"x": 1}]}}) is True
    assert mod._contains_rows({"plain": "value"}) is False

    missing_expected = mod._run_artifact_convention_check(
        mod.V569_ARTIFACTS[4], tmp_path / "missing-exp6569.json", {}
    )
    missing_unexpected = mod._run_artifact_convention_check(
        mod.V569_ARTIFACTS[0], tmp_path / "missing-exp6565.json", {}
    )
    assert missing_expected["status"] == "EXPECTED_MISSING"
    assert missing_unexpected["status"] == "CANNOT_DETERMINE"

    blocked_path = tmp_path / "blocked.json"
    blocked_path.write_text("{}", encoding="utf-8")
    blocked_without_reason = mod._run_artifact_convention_check(
        mod.V569_ARTIFACTS[0], blocked_path, {"status": "blocked"}
    )
    aggregate_only = mod._run_artifact_convention_check(
        mod.V569_ARTIFACTS[0], blocked_path, {"status": "complete"}
    )
    checkable = mod._run_artifact_convention_check(
        mod.V569_ARTIFACTS[0], blocked_path, {"status": "complete", "per_unit_rows": [{}]}
    )
    assert blocked_without_reason["status"] == "BLOCKED_WITHOUT_DIAGNOSTIC"
    assert aggregate_only["status"] == "AGGREGATE_ONLY"
    assert checkable["status"] == "CHECKABLE"

    bad_live = _fake_live_checks()["exp6565"]
    bad_live["artifact_convention"]["status"] = "AGGREGATE_ONLY"
    unusable = mod._artifact_outcome(
        mod.V569_ARTIFACTS[0],
        path=REPO / mod.V569_ARTIFACTS[0].relative_path,
        payload=json.loads((REPO / mod.V569_ARTIFACTS[0].relative_path).read_text()),
        live_result=bad_live,
    )
    assert unusable["disposition"] == "unusable_live_verifier_result"
    changed_missing = mod._artifact_outcome(
        mod.V569_ARTIFACTS[4],
        path=REPO / mod.V569_ARTIFACTS[0].relative_path,
        payload={"status": "complete"},
        live_result=_fake_live_checks()["exp6569"],
    )
    assert changed_missing["disposition"] == "frozen_missing_path_changed"

    retired = mod._retired_experiment_ids(
        {
            "retired_experiments": "skip",
            "retired_extras": [None],
            "retired": [
                {"id": "retired-a", "experiment_id": "retired-b", "experiment_ids": ["retired-c"]}
            ],
        }
    )
    assert retired == {"retired-a", "retired-b", "retired-c"}
    assert mod._retired_dependency_ids(
        {
            "tasks": [
                "skip",
                {"requires": ["retired-a"]},
                {"gated_on": [{"upstream": "retired-b"}]},
            ]
        },
        retired,
    ) == {"retired-a", "retired-b"}
    assert mod._task_manifest({"tasks": ["skip"]}, {}) == []
    assert (
        mod._gate_contract_rows(
            {"tasks": ["skip", {"id": "task", "gated_on": ["skip"]}]}, {}, set()
        )
        == []
    )
    assert mod._scope_class("exp0000-other") == "prior_failed_scope"
    bad_prior_rows = mod._prior_failure_rows(
        {"tasks": ["skip", {"id": "task", "prior_failures": ["bad"]}]}, set(), set()
    )
    assert bad_prior_rows[0]["complete_prior_failure_contract"] is False

    candidate = deepcopy(artifact)
    candidate["field_principles"] = {}
    candidate["v569_artifact_eligibility_rows"][0]["expected_path"] = "wrong.json"
    alias_path = REPO / mod.V569_ARTIFACTS[1].relative_path
    candidate["v569_artifact_eligibility_rows"][0]["resolved_path"] = str(alias_path)
    candidate["v569_artifact_eligibility_rows"][0]["sha256"] = mod.sha256_file(alias_path)
    candidate["v569_artifact_eligibility_rows"][0]["eligible_for_v570_contract"] = False
    candidate["v570_gate_contract_rows"][0]["upstream_in_active_roadmap"] = False
    candidate["v570_gate_contract_rows"][0]["retired_upstream"] = True
    candidate["prior_failure_and_retirement_rows"][0]["complete_prior_failure_contract"] = False
    candidate["prior_failure_and_retirement_rows"][0]["mechanical_repeat_retirement_rule"] = False
    candidate["prior_failure_and_retirement_rows"][0]["retired_dependency_chain"] = True
    candidate["model_arc_and_hardware_boundary"]["legacy_model_policy"][
        "legacy_substitution_allowed"
    ] = True
    candidate["model_arc_and_hardware_boundary"]["extraction_boundary"][
        "retired_constraintir_reuse_allowed"
    ] = True
    candidate["model_arc_and_hardware_boundary"]["hardware_boundary"][
        "unchanged_hardware_command_allowed"
    ] = True
    _with_checksum(candidate)
    errors = mod.validate_artifact(candidate)
    for expected in (
        "field_principles must cover required fields",
        "exact path mismatch for exp6565",
        "V569 artifact path alias for exp6565",
        "exp6565 readiness hides an ineligible row",
        "gate upstream is outside active roadmap",
        "gate references retired upstream",
        "prior failure row is incomplete",
        "prior failure row lacks repeat-retirement rule",
        "prior failure row reuses a retired dependency",
        "legacy model substitution opened",
        "retired ConstraintIR reuse opened",
        "unchanged hardware command opened",
    ):
        assert expected in errors
