"""Tests for Exp6565 V569 evidence and retirement contract.

Spec refs: REQ-REPORT-6565, SCENARIO-REPORT-6565-IMPORT,
SCENARIO-REPORT-6565-LIVE-REPLAY, SCENARIO-REPORT-6565-GATES,
SCENARIO-REPORT-6565-PRIOR-FAILURE,
SCENARIO-REPORT-6565-MODEL-ARC-HARDWARE,
SCENARIO-REPORT-6565-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6565_v569_evidence_and_retirement_contract as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": "focused-exp6565", "exit_code": 0}]


def _fake_live_clean() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for artifact in mod.V568_ARTIFACTS:
        results[artifact.exp_id] = {
            "adversarial": {
                "command": f"adversarial {artifact.exp_id}",
                "exit_code": 0,
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
                "duration_s": 0.01,
            },
            "row_consistency": {
                "command": f"row-lint {artifact.exp_id}",
                "exit_code": 0,
                "status": "ok",
                "findings": [],
                "duration_s": 0.01,
            },
        }
    return results


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    """REQ-REPORT-6565: build from checked-in V568 artifacts with fake live checks."""

    return mod.build_artifact(
        repo_root=REPO,
        result_path=Path("/tmp/experiment_6565_test_result.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_live_clean(),
        run_date="20260823",
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = mod.reproducibility_checksum(payload)
    return payload


def test_req_report_6565_spec_declares_required_contract() -> None:
    """REQ-REPORT-6565: OpenSpec owns the V569 evidence contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6565") : text.index("REQ-REPORT-6561")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-6565-IMPORT",
        "SCENARIO-REPORT-6565-LIVE-REPLAY",
        "SCENARIO-REPORT-6565-GATES",
        "SCENARIO-REPORT-6565-PRIOR-FAILURE",
        "SCENARIO-REPORT-6565-MODEL-ARC-HARDWARE",
        "SCENARIO-REPORT-6565-ATOMIC",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6565_import_rows_classify_v568_boundary(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6565-IMPORT: V568 rows are content addressed and classified."""

    assert mod.validate_artifact(artifact) == []
    rows = {row["exp_id"]: row for row in artifact["v568_artifact_eligibility_rows"]}

    assert list(rows) == [artifact.exp_id for artifact in mod.V568_ARTIFACTS]
    assert rows["exp6561"]["eligible_for_v569_contract"] is True
    assert rows["exp6561"]["disposition"] == "usable_contract_with_stamped_duration_caution"
    assert rows["exp6561"]["stamped_live_flag_disagreement"] is True
    assert {flag["kind"] for flag in rows["exp6561"]["stamped_flags"]} >= {"DURATION_TOO_SHORT"}

    assert rows["exp6562"]["disposition"] == "disqualified_saturation_science"
    assert rows["exp6562"]["failed_scope"] == "constraint_saturation"
    assert rows["exp6562"]["extends_exp6556_saturation_headline"] is False
    assert rows["exp6562"]["verdict_class"] == "disqualified"

    assert rows["exp6563"]["disposition"] == "clean_null_production_evidence"
    assert rows["exp6563"]["production_adapter_default_off"] is True
    assert rows["exp6563"]["promotion_candidate"] is False

    assert rows["exp6564"]["disposition"] == "clean_null_nfr01_evidence"
    assert rows["exp6564"]["nfr01_passed"] is False
    assert rows["exp6564"]["measured_speedup_vs_requirement"] == pytest.approx(0.764013447)

    for row in rows.values():
        assert row["expected_path"].startswith("results/experiment_656")
        assert row["sha256"].startswith("sha256:")
        assert row["live_verifier_exit_code"] == 0
        assert row["row_consistency_status"] == "ok"
        assert row["eligible_for_v569_contract"] is True


def test_scenario_report_6565_live_replay_rows_resolve_stamped_state(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6565-LIVE-REPLAY: live rows record current verifier replay."""

    rows = {row["exp_id"]: row for row in artifact["live_verifier_and_duration_rows"]}

    assert set(rows) == {artifact.exp_id for artifact in mod.V568_ARTIFACTS}
    exp6561 = rows["exp6561"]
    assert exp6561["stamped_flag_count"] >= 1
    assert exp6561["live_flag_count"] == 0
    assert exp6561["stamped_live_flag_disagreement"] is True
    assert exp6561["reason"] == "stamped_duration_flag_recorded_live_replay_clean"

    for row in rows.values():
        assert row["adversarial_verifier_version"]["sha256"].startswith("sha256:")
        assert row["row_lint_version"]["sha256"].startswith("sha256:")
        assert row["live_verifier_command"].startswith("adversarial")
        assert row["row_consistency_command"].startswith("row-lint")
        assert row["artifact_duration_s"] is None or row["artifact_duration_s"] >= 0.0


def test_scenario_report_6565_gates_and_prior_failures_close(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6565-GATES/PRIOR-FAILURE: active roadmap contracts close."""

    gate_rows = artifact["v569_gate_contract_rows"]
    gate_key = {(row["task_id"], row["upstream"], row["artifact_field"]) for row in gate_rows}

    assert len(gate_rows) == 5
    assert all(row["upstream_in_active_roadmap"] for row in gate_rows)
    assert all(row["artifact_field_declared_by_upstream"] for row in gate_rows)
    assert all(row["retired_upstream"] is False for row in gate_rows)
    assert (
        "exp6567-sequential-flagship-gguf-admission",
        "exp6565-v569-evidence-and-retirement-contract",
        "v569_evidence_contract_ready_score",
    ) in gate_key
    assert (
        "exp6569-source-span-proof-obligation-extractor",
        "exp6568-immutable-source-span-claim-stream",
        "immutable_live_claim_stream_ready_score",
    ) in gate_key

    prior_rows = artifact["prior_failure_and_retirement_rows"]
    assert len(prior_rows) == 9
    assert all(row["complete_prior_failure_contract"] for row in prior_rows)
    assert all(row["changed_mechanism"] for row in prior_rows)
    assert all(row["retire_if_same_verdict"] is True for row in prior_rows)
    assert all(row["retired_dependency_chain"] is False for row in prior_rows)
    assert artifact["gate_check_summary"]["task_field_gate_contract_closed"] is True
    assert artifact["gate_check_summary"]["prior_failure_retirement_contract_closed"] is True


def test_scenario_report_6565_model_arc_hardware_and_rust_fusion_boundary(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-6565-MODEL-ARC-HARDWARE: boundaries and retirement are frozen."""

    boundary = artifact["model_arc_and_hardware_boundary"]

    assert boundary["MODEL_SPECS"] == list(mod.MANDATED_MODEL_IDS)
    assert boundary["legacy_model_policy"]["legacy_smoke_models_can_support_headline"] is False
    assert boundary["arc_boundary"]["no_game_or_level_solve_claim"] is True
    assert boundary["hardware_boundary"]["exp6565_hardware_command_count"] == 0
    assert boundary["hardware_boundary"]["unchanged_board_command_allowed"] is False
    assert boundary["production_boundary"]["production_adapter_default_off"] is True
    assert boundary["rust_fusion_boundary"]["active_roadmap_has_exp6574"] is False
    assert boundary["rust_fusion_boundary"]["proposed_exp6574_materially_different"] is True
    assert boundary["rust_fusion_boundary"]["retire_on_repeated_no_benefit_or_nfr01_miss"] is True
    assert artifact["rust_fusion_reopen_ready_score"] == 1.0


def test_scenario_report_6565_schema_validation_and_attack_matrix(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6565-ATOMIC: output is atomic, checksummed, and defensive."""

    result_path = tmp_path / "exp6565.json"
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_live_clean(),
        run_date="20260823",
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert loaded["reproducibility_checksum"] == written["reproducibility_checksum"]
    assert written["status"] == "complete_v569_evidence_and_retirement_contract_ready"
    assert written["honest_verdict"].startswith("complete_")
    assert written["verdict_class"] is None
    assert written["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert written["verifier_is_oracle"] is True
    assert written["v569_evidence_contract_ready_score"] == 1.0
    assert written["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(written)
    assert set(written["field_provenance"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(written["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)

    classification = adversarial_verify._classify_inference_substrate(written)
    report = adversarial_verify.verify_artifact(result_path)
    assert classification["kind"] == adversarial_verify.SUBSTRATE_KIND_NO_LLM
    assert classification["matched_value"] == mod.INFERENCE_SUBSTRATE
    assert report["flag_count"] == 0

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
            lambda data: data["v568_artifact_eligibility_rows"][3].__setitem__(
                "sha256", "sha256:alias"
            ),
            "V568 artifact hash alias",
        ),
        (
            lambda data: data["v569_gate_contract_rows"][0].__setitem__(
                "artifact_field_declared_by_upstream", False
            ),
            "gate contract has undeclared field",
        ),
        (
            lambda data: data["prior_failure_and_retirement_rows"][0].pop("addressed_by"),
            "prior failure row missing required fields",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"].__setitem__(
                "MODEL_SPECS", ["legacy"]
            ),
            "mandated GGUF model identities changed",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"]["arc_boundary"].__setitem__(
                "no_game_or_level_solve_claim", False
            ),
            "ARC solve boundary opened",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"]["hardware_boundary"].__setitem__(
                "exp6565_hardware_command_count", 1
            ),
            "hardware command boundary violated",
        ),
        (
            lambda data: data["model_arc_and_hardware_boundary"][
                "rust_fusion_boundary"
            ].__setitem__("proposed_exp6574_materially_different", False),
            "rust fusion reopen score must derive from changed workload and retirement rule",
        ),
        (
            lambda data: data["gate_check_summary"].__setitem__("failed_checks", ["forced"]),
            "ready score cannot be open with failed checks",
        ),
        (
            lambda data: data["aggregate_row_recomputation"].__setitem__(
                "v569_evidence_contract_ready_from_rows", False
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
    ]
    for mutate, expected in mutations:
        candidate = deepcopy(written)
        mutate(candidate)
        _with_checksum(candidate)
        assert any(expected in error for error in mod.validate_artifact(candidate))

    bad_checksum = deepcopy(written)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)


def test_scenario_report_6565_missing_input_blocks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6565-IMPORT: missing input closes blocked, not null."""

    paths = mod.default_v568_paths(REPO)
    paths["exp6562"] = tmp_path / "missing-exp6562.json"
    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        check_results=_fake_live_clean(),
        artifact_paths=paths,
        run_date="20260823",
    )

    assert blocked["status"] == "blocked_v569_evidence_contract_missing_inputs"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["v569_evidence_contract_ready_score"] == 0.0
    assert blocked["rust_fusion_reopen_ready_score"] == 1.0
    assert "exp6562_input_exists" in blocked["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(blocked) == []


def test_scenario_report_6565_helper_edges_are_deterministic(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6565: helper edge cases fail closed."""

    assert mod.sha256_file(None) == "missing"
    assert mod.sha256_file(Path("/tmp/definitely-missing-exp6565-file")) == "missing"
    assert mod._read_json(Path("/tmp/definitely-missing-exp6565-file.json")) == {}  # noqa: SLF001
    assert mod.load_json(Path("/tmp/definitely-missing-exp6565-file.json")) == {}
    assert mod.default_v568_paths(REPO)["exp6561"].name.startswith("experiment_6561")
    assert mod._coerce_closed_verdict_class(None) is None  # noqa: SLF001
    assert mod._coerce_closed_verdict_class("null") == "null"  # noqa: SLF001
    assert mod._coerce_closed_verdict_class("weird") == "disqualified"  # noqa: SLF001
    assert mod._artifact_duration_reason("exp6561", [{"kind": "DURATION_TOO_SHORT"}], []) == (
        "stamped_duration_flag_recorded_live_replay_clean"
    )
    assert mod._artifact_duration_reason("exp6561", [], [{"severity": "critical"}]) == (  # noqa: SLF001
        "live_verifier_critical_flags_recorded"
    )
    assert mod._artifact_duration_reason("exp6562", [], []) == "live_replay_clean"  # noqa: SLF001
    assert (  # noqa: SLF001
        mod._prior_failure_scope_class("exp6564-rust-pyo3-safety-net-nfr01")
        == "safety_net_production_or_nfr01_null"
    )
    assert mod._speedup_from_payload({"aggregate_row_recomputation": {"x": 1}}) is None  # noqa: SLF001
    assert mod._speedup_from_payload({}) is None  # noqa: SLF001
    assert mod._artifact_outcome(mod.V568_ARTIFACTS[0], {}, []) == {  # noqa: SLF001
        "disposition": "missing_input",
        "failed_scope": "missing_v568_artifact",
        "eligible": False,
        "reason": "exp6561_input_exists",
    }
    assert (
        mod._artifact_outcome(  # noqa: SLF001
            mod.V568_ARTIFACTS[0], {"verdict_class": None}, [{"severity": "critical"}]
        )["disposition"]
        == "not_imported_live_verifier_critical"
    )

    fields = mod._parse_required_fields(  # noqa: SLF001
        "REQUIRED ARTIFACT FIELDS:\n  status:\n    principle: x\nRun command:"
    )
    assert fields == {"status"}

    retired = mod._retired_experiment_ids(  # noqa: SLF001
        {
            "retired": [None, {"id": "retired-a", "experiment_ids": ["retired-b"]}],
            "retired_experiments": "skip",
        }
    )
    assert retired == {"retired-a", "retired-b"}

    assert mod._requires_retired_ids(  # noqa: SLF001
        {
            "tasks": [
                "skip",
                {"id": "task-a", "requires": ["retired-a"]},
                {"id": "task-b", "gated_on": [{"upstream": "retired-b"}]},
            ]
        },
        {"retired-a", "retired-b"},
    ) == {"retired-a", "retired-b"}

    assert (
        mod._gate_contract_rows(  # noqa: SLF001
            {"tasks": ["skip", {"id": "task", "gated_on": ["skip-gate"]}]},
            {"upstream": {"field"}},
            {"upstream"},
        )
        == []
    )

    prior_rows = mod._prior_failure_and_retirement_rows(  # noqa: SLF001
        {
            "tasks": [
                "skip",
                {"id": "bad", "prior_failures": ["not-a-row"]},
                {"id": "task", "prior_failures": [{"experiment_id": "old"}]},
            ]
        },
        {"old"},
        {"old"},
    )
    assert prior_rows[0]["complete_prior_failure_contract"] is False
    assert prior_rows[1]["complete_prior_failure_contract"] is False
    assert prior_rows[1]["retired_dependency_chain"] is True

    summary = mod._gate_check_summary(  # noqa: SLF001
        rows=[
            {
                "exp_id": "exp6561",
                "exists": False,
                "expected_path": "missing.json",
                "eligible_for_v569_contract": False,
            }
        ],
        gate_rows=[
            {
                "task_id": "task",
                "upstream": "retired-upstream",
                "artifact_field": "bad_field",
                "upstream_in_active_roadmap": False,
                "artifact_field_declared_by_upstream": False,
                "retired_upstream": True,
            }
        ],
        prior_rows=[{"complete_prior_failure_contract": False, "retired_dependency_chain": True}],
        boundary={"all_boundary_checks_passed": False},
        protected={"all_unchanged": False, "changed_paths": ["research-roadmap.yaml"]},
    )
    assert {
        "exp6561_input_exists",
        "exp6561_contract_eligible",
        "gate_upstream_in_active_roadmap",
        "gate_artifact_field_declared",
        "gate_retired_upstream",
        "prior_failure_contract_complete",
        "prior_failure_retired_dependency_chain_absent",
        "model_arc_hardware_boundary_closed",
        "protected_files_unchanged",
    } <= set(summary["failed_checks"])

    assert mod._status_and_verdict(True, False, []) == (  # noqa: SLF001
        "complete_v569_evidence_and_retirement_contract_ready",
        "complete_v569_evidence_and_retirement_contract_ready: V568 artifacts are content-addressed; Exp6562 is disqualified science; Exp6563 and Exp6564 are clean nulls; V569 gate, failure, model, ARC, hardware, Rust-fusion, and protected-file contracts close",
        None,
    )
    assert mod._status_and_verdict(False, True, ["missing"]) == (  # noqa: SLF001
        "blocked_v569_evidence_contract_missing_inputs",
        "blocked_v569_evidence_contract_missing_inputs: required V568 input artifact is missing; failed checks are recorded",
        "blocked",
    )
    assert mod._status_and_verdict(False, False, ["gate"]) == (  # noqa: SLF001
        "partial_v569_evidence_and_retirement_contract",
        "partial_v569_evidence_and_retirement_contract: usable V568 evidence exists but one or more gate, failure, model, ARC, hardware, Rust-fusion, or protected-file checks failed",
        "partial",
    )
    assert mod._status_and_verdict(False, False, []) == (  # noqa: SLF001
        "blocked_v569_evidence_contract",
        "blocked_v569_evidence_contract: no usable V568 input set was available",
        "blocked",
    )

    malformed = tmp_path / "not-json.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._read_json(malformed) == {}  # noqa: SLF001

    more_edges = deepcopy(artifact)
    more_edges["v568_artifact_eligibility_rows"][0]["eligible_for_v569_contract"] = False
    more_edges["v569_gate_contract_rows"] = [
        "not-a-row",
        {
            "artifact_field_declared_by_upstream": True,
            "upstream_in_active_roadmap": False,
            "retired_upstream": True,
        },
    ]
    more_edges["prior_failure_and_retirement_rows"] = [
        "not-a-row",
        {
            "experiment_id": "old",
            "verdict": "x",
            "addressed_by": "",
            "retire_if_same_verdict": False,
            "complete_prior_failure_contract": True,
            "changed_mechanism": False,
            "mechanical_repeat_retirement_rule": False,
            "retired_dependency_chain": True,
        },
    ]
    more_edges["model_arc_and_hardware_boundary"]["legacy_model_policy"][
        "legacy_smoke_models_can_support_headline"
    ] = True
    more_edges["model_arc_and_hardware_boundary"]["hardware_boundary"][
        "unchanged_board_command_allowed"
    ] = True
    _with_checksum(more_edges)
    errors = mod.validate_artifact(more_edges)
    assert "exp6561 readiness score hides ineligible row" in errors
    assert "gate contract row must be a mapping" in errors
    assert "gate contract has out-of-roadmap upstream" in errors
    assert "gate contract has retired upstream" in errors
    assert "prior failure row must be a mapping" in errors
    assert "prior failure row lacks changed mechanism" in errors
    assert "prior failure row lacks mechanical repeat-retirement rule" in errors
    assert "prior failure row uses retired dependency chain" in errors
    assert "legacy-model substitution opened" in errors
    assert "unchanged hardware command boundary opened" in errors
