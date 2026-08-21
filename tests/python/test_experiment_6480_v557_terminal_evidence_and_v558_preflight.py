"""Tests for Exp6480 V557 terminal evidence and V558 preflight.

Spec refs: REQ-INFRA-6480, SCENARIO-INFRA-6480-1,
SCENARIO-INFRA-6480-2, SCENARIO-INFRA-6480-3,
SCENARIO-INFRA-6480-4, SCENARIO-INFRA-6480-5,
SCENARIO-INFRA-6480-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6480_v557_terminal_evidence_and_v558_preflight as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_ARTIFACT_CACHE: dict[str, Any] | None = None


def _artifact() -> dict[str, Any]:
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        _ARTIFACT_CACHE = mod.build_artifact(
            repo_root=REPO,
            result_path=Path("/tmp/experiment_6480_test_result.json"),
            write=False,
            duration_s=1.0,
            tests_run=[{"command": "focused", "exit_code": 0}],
        )
    return deepcopy(_ARTIFACT_CACHE)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_infra_6480_spec_declares_required_contract() -> None:
    """REQ-INFRA-6480: OpenSpec owns the Exp6480 report contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6480") :]

    for marker in (
        "SCENARIO-INFRA-6480-1",
        "SCENARIO-INFRA-6480-2",
        "SCENARIO-INFRA-6480-3",
        "SCENARIO-INFRA-6480-4",
        "SCENARIO-INFRA-6480-5",
        "SCENARIO-INFRA-6480-6",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_infra_6480_terminal_rows_account_for_v557() -> None:
    """SCENARIO-INFRA-6480-1: all seven V557 tasks have terminal rows."""

    artifact = _artifact()
    rows = {row["task_id"]: row for row in artifact["v557_terminal_rows"]}

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_v557_terminal_evidence_frozen"
    assert [row["task_id"] for row in artifact["v557_terminal_rows"]] == [
        task.task_id for task in mod.EXPECTED_V557_TASKS
    ]
    assert artifact["artifact_hash_manifest"]["expected_count"] == 7
    assert artifact["artifact_hash_manifest"]["present_count"] == 7
    assert artifact["artifact_hash_manifest"]["zero_byte_count"] == 0
    assert artifact["artifact_hash_manifest"]["absent_count"] == 0

    exp6479 = rows["exp6479-verify-repair-factor-cache-shadow-adapter"]
    assert exp6479["artifact_state"] == "complete"
    assert exp6479["bytes"] > 0
    assert exp6479["sha256"].startswith("sha256:")
    assert exp6479["readiness_fields"]["factor_cache_shadow_adapter_ready_score"] == 1.0
    assert exp6479["adversarial_status"]["status"] in {
        "declared_clean",
        "not_declared_in_artifact",
    }

    for row in artifact["v557_terminal_rows"]:
        assert row["path"].startswith("results/experiment_")
        assert row["honest_verdict"]
        assert row["gate_diagnostics"]["check"]
        assert row["adversarial_status"]["status"]


def test_scenario_infra_6480_artifact_states_stay_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6480-2: terminal artifact states are not collapsed."""

    complete = tmp_path / "complete.json"
    _write_json(complete, {"status": "success", "honest_verdict": "success: fixture"})
    blocked = tmp_path / "blocked.json"
    _write_json(
        blocked,
        {
            "status": "blocked_gate_check_failed",
            "honest_verdict": "complete_blocked: fixture",
            "gate_check_summary": {
                "failed_checks": [
                    {
                        "check": "upstream",
                        "expected": 1,
                        "observed": 0,
                        "evidence_path": "blocked.json",
                    }
                ]
            },
        },
    )
    null = tmp_path / "null.json"
    _write_json(null, {"status": "complete_null_result", "honest_verdict": "complete_null: no"})
    zero = tmp_path / "zero.json"
    zero.touch()
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "EXPECTED_V557_TASKS",
        (
            mod.ExpectedTask("exp-complete", "expC", Path("complete.json"), "complete"),
            mod.ExpectedTask("exp-blocked", "expB", Path("blocked.json"), "blocked"),
            mod.ExpectedTask("exp-null", "expN", Path("null.json"), "null"),
            mod.ExpectedTask("exp-zero", "expZ", Path("zero.json"), "zero"),
            mod.ExpectedTask("exp-malformed", "expM", Path("malformed.json"), "bad"),
            mod.ExpectedTask("exp-missing", "expX", Path("missing.json"), "missing"),
        ),
    )

    rows, payloads = mod.v557_terminal_rows(tmp_path)
    by_id = {row["task_id"]: row for row in rows}

    assert set(payloads) == {"expC", "expB", "expN"}
    assert by_id["exp-complete"]["artifact_state"] == "complete"
    assert by_id["exp-blocked"]["artifact_state"] == "blocked"
    assert by_id["exp-null"]["artifact_state"] == "null"
    assert by_id["exp-zero"]["artifact_state"] == "zero_byte"
    assert by_id["exp-malformed"]["artifact_state"] == "malformed"
    assert by_id["exp-missing"]["artifact_state"] == "missing"
    assert by_id["exp-blocked"]["gate_diagnostics"]["expected"] == 1
    assert mod.artifact_hash_manifest(rows)["zero_byte_paths"] == ["zero.json"]
    assert mod.sha256_file(tmp_path / "missing.json") is None


def test_scenario_infra_6480_retirement_and_exact_energy_boundaries() -> None:
    """SCENARIO-INFRA-6480-3 and 5: branch boundaries are narrow."""

    artifact = _artifact()
    retirement = artifact["retirement_boundary_rows"][0]
    exact = artifact["exact_energy_evidence_boundary"]

    assert retirement["lineage_id"] == "Exp6463"
    assert retirement["disposition"] == "retired_for_held_evidence"
    assert retirement["may_create_new_prospective_lineage"] is True
    assert retirement["may_reuse_exp6463_held_evidence"] is False
    assert retirement["held_unit_count"] == 36
    assert retirement["held_units_with_both_precommit_proofs"] == 0
    assert {
        row["check"]: row["observed_value"] for row in retirement["failed_gate_rows"]
    } == {
        "all_held_units_have_label_precommit_proof": False,
        "all_held_units_have_membership_precommit_proof": False,
        "no_missing_or_posthoc_held_rows": False,
    }

    assert exact["exact_record_ready_score"] == 1.0
    assert exact["held_exact_energy_selection_ready_score"] == 1.0
    assert exact["finite_no_llm_unit_seed_count"] == 24
    assert exact["local_sota_extension_claimed"] is False
    assert exact["local_sota_output_evidence_status"] == "not_supported_by_exp6478"
    assert exact["boundary_statement"].endswith("does not extend to local-SOTA outputs.")


def test_scenario_infra_6480_narrow_readiness_scores() -> None:
    """SCENARIO-INFRA-6480-4: readiness uses only allowed upstream gates."""

    artifact = _artifact()
    payloads = mod.load_upstream_payloads(REPO)

    assert artifact["v557_factor_cache_ready_score"] == 1.0
    assert artifact["v557_arc_shield_ready_score"] == 1.0
    assert mod.factor_cache_ready_boundary(payloads["exp6479"])["score"] == 1.0
    assert mod.arc_shield_ready_boundary(payloads["exp6471"])["score"] == 1.0

    factor_bad = deepcopy(payloads["exp6479"])
    factor_bad["default_off_compatibility_rows"]["all_public_outputs_match"] = False
    assert mod.factor_cache_ready_boundary(factor_bad)["score"] == 0.0

    factor_bad = deepcopy(payloads["exp6479"])
    factor_bad["tests_run"] = {"exit_codes": {"focused": 1}}
    assert mod.factor_cache_ready_boundary(factor_bad)["score"] == 0.0

    arc_bad = deepcopy(payloads["exp6471"])
    arc_bad["current_adversarial_findings"]["critical_count"] = 1
    assert mod.arc_shield_ready_boundary(arc_bad)["score"] == 0.0

    arc_bad = deepcopy(payloads["exp6471"])
    arc_bad["no_solve_claim"] = False
    assert mod.arc_shield_ready_boundary(arc_bad)["score"] == 0.0


def test_scenario_infra_6480_schema_write_and_validation_edges(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6480-6: validation rejects activation and drift."""

    artifact = _artifact()
    out = tmp_path / "artifact.json"
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=out,
        write=True,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert out.is_file()
    assert mod.load_json(out) == written
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["staged_queue_validation_performed"] is False
    assert artifact["roadmap_activation_performed"] is False
    assert artifact["unrelated_branch_gate_count"] == 0
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["rows"] == artifact["per_unit_rows"]

    mutations = [
        ("missing required fields", lambda data: data.pop("status")),
        (
            "staged_queue_validation_performed must be false",
            lambda data: data.__setitem__("staged_queue_validation_performed", True),
        ),
        (
            "roadmap_activation_performed must be false",
            lambda data: data.__setitem__("roadmap_activation_performed", True),
        ),
        (
            "unrelated_branch_gate_count must be 0",
            lambda data: data.__setitem__("unrelated_branch_gate_count", 1),
        ),
        (
            "V557 terminal row count mismatch",
            lambda data: data.__setitem__(
                "v557_terminal_rows", data["v557_terminal_rows"][:-1]
            ),
        ),
        (
            "v557_factor_cache_ready_score mismatch",
            lambda data: data.__setitem__("v557_factor_cache_ready_score", 0.0),
        ),
        (
            "v557_arc_shield_ready_score mismatch",
            lambda data: data.__setitem__("v557_arc_shield_ready_score", 0.0),
        ),
        (
            "inference_substrate mismatch",
            lambda data: data.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be true",
            lambda data: data.__setitem__("verifier_is_oracle", False),
        ),
        (
            "honest_verdict lacks required terminal prefix",
            lambda data: data.__setitem__("honest_verdict", "done"),
        ),
        (
            "protected files changed",
            lambda data: data["protected_files_unchanged"].__setitem__("unchanged", False),
        ),
        (
            "field_principles must cover exactly required fields",
            lambda data: data.__setitem__("field_principles", {}),
        ),
        (
            "field_provenance must cover exactly required fields",
            lambda data: data.__setitem__("field_provenance", {}),
        ),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        bad = _with_checksum(bad)
        assert any(expected in error for error in mod.validate_artifact(bad)), expected

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)


def test_scenario_infra_6480_helper_edges_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6480: helper edge cases stay explicit and closed."""

    artifact = _artifact()

    assert mod._status_text(None) == ""
    assert mod.tests_run_receipts(None)[0]["exit_code"] is None
    assert mod._false_gate_rows({"failed_gates": ["x"]}, "evidence") == []
    assert mod._git_output(REPO, ["not-a-real-git-subcommand"]).startswith("git_failed:")
    assert mod.validate_artifact(tmp_path / "missing.json")[0].startswith("unloadable artifact")

    declared = mod._adversarial_status(
        {"current_adversarial_findings": {"critical_count": 1, "flag_count": 2}},
        Path("artifact.json"),
    )
    assert declared["status"] == "declared_findings"

    diagnostic = mod.normalize_gate_diagnostics(
        {
            "gate_check_summary": {
                "checks": {"first": True, "second": False},
                "missing_evidence_path": "evidence.json",
            }
        },
        relative_path=Path("artifact.json"),
        artifact_state="complete",
    )
    assert diagnostic == {
        "check": "second",
        "expected": True,
        "observed": False,
        "evidence_path": "evidence.json",
    }

    with monkeypatch.context() as patch:
        patch.setattr(mod, "validate_artifact", lambda _artifact: ["forced validation error"])
        with pytest.raises(ValueError, match="forced validation error"):
            mod.build_artifact(repo_root=REPO, write=False, duration_s=1.0)

    bad = deepcopy(artifact)
    bad["v557_terminal_rows"][0]["task_id"] = "wrong"
    assert "V557 terminal row ids mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["v557_terminal_rows"][0]["artifact_disposition"] = ""
    assert "V557 terminal row missing artifact disposition" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["v557_terminal_rows"][0]["artifact_state"] = "blocked"
    bad["v557_terminal_rows"][0]["gate_diagnostics"] = {"check": "", "expected": 1}
    assert "blocked row missing normalized gate diagnostic" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["artifact_hash_manifest"]["expected_count"] = 0
    assert "artifact_hash_manifest expected_count mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["exact_energy_evidence_boundary"]["local_sota_extension_claimed"] = True
    assert "exact energy boundary must not claim local-SOTA extension" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["gate_check_summary"] = []
    assert "gate_check_summary must be a mapping" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["gate_check_summary"]["acceptance_gates"] = []
    assert "acceptance gates missing" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["gate_check_summary"]["acceptance_gates"][0]["passed"] = False
    assert "all seven V557 task IDs must be accounted" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["gate_check_summary"]["acceptance_gates"][1]["passed"] = False
    assert "queue and activation boundary must pass" in mod.validate_artifact(
        _with_checksum(bad)
    )
