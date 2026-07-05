"""Tests for Exp 5276 memory-assisted verifier-dose gated pilot.

Spec refs: REQ-VERIFY-5276, SCENARIO-VERIFY-5276.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5276_memory_assisted_verifier_dose_gated_v482 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _blocked_artifacts() -> dict[str, dict[str, Any]]:
    artifacts = mod.load_upstream_artifacts(REPO)
    artifacts["exp5271"] = {
        **artifacts["exp5271"],
        "telemetry_harness_ready": False,
        "honest_verdict": {"value": "blocked_no_live_sota_receipts"},
    }
    return artifacts


def test_req_verify_5276_spec_declares_memory_assisted_dose_contract() -> None:
    """REQ-VERIFY-5276: OpenSpec anchors the gated memory/dose pilot."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5276") :]

    for marker in (
        "REQ-VERIFY-5276",
        "SCENARIO-VERIFY-5276",
        str(mod.RESULT_RELATIVE_PATH),
        "live_llm_inference_local_gguf_sota",
        "telemetry_harness_ready=true",
        "memory_decision_history_ready=true",
        "always-full-verifier baseline",
        "no-memory scheduler baseline",
        "memory MAY choose whether a full verifier dose is needed",
        "Governed memory SHALL NOT inject the answer",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in section
    normalized_section = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5276_preconditions_block_when_upstream_gate_closed() -> None:
    """REQ-VERIFY-5276: closed gates write blocked/unmeasured, not fake metrics."""

    artifact = mod.build_result_artifact(
        root=REPO,
        upstream_artifacts=_blocked_artifacts(),
        commands_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert "unmeasured" in artifact["honest_verdict"]["value"]
    assert artifact["memory_verifier_dose_ready"]["value"] is False
    assert artifact["calls_avoided_rate"]["value"] == 0.0
    assert artifact["decision_quality_delta"]["value"] == 0.0
    assert artifact["unsafe_false_accepts"]["value"] == 0
    assert artifact["rollback_exercised"]["value"] is False
    assert artifact["memory_scope_violations_blocked"]["value"] == 0
    assert artifact["pilot_rows"] == []
    assert artifact["preconditions_checked"]["value"]["all_gates_ready"] is False
    assert (
        "exp5271.telemetry_harness_ready" in artifact["preconditions_checked"]["value"]["blockers"]
    )
    mod.validate_artifact(artifact)


def test_req_verify_5276_model_specs_preserve_live_sota_headline_roles() -> None:
    """REQ-VERIFY-5276: headline model specs come only from mandated SOTA GGUF receipts."""

    artifacts = mod.load_upstream_artifacts(REPO)
    model_specs = mod.extract_model_specs(artifacts["exp5271"])

    assert set(model_specs) == {"flagship_moe", "flagship_dense", "middle_moe"}
    assert model_specs["flagship_moe"]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert model_specs["flagship_dense"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert model_specs["middle_moe"]["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert all(spec["headline_role"] is True for spec in model_specs.values())
    assert all("Qwen3.5-0.8B" not in json.dumps(spec) for spec in model_specs.values())


def test_scenario_verify_5276_memory_is_allocation_feature_not_answer_source() -> None:
    """SCENARIO-VERIFY-5276: governed memory cannot inject selected decisions directly."""

    artifacts = mod.load_upstream_artifacts(REPO)
    rows = mod.build_pilot_rows(root=REPO, upstream_artifacts=artifacts)
    mutated = deepcopy(artifacts)
    for row in mutated["exp5275"]["governance_rows"]:
        if row["fixture_kind"] == "promotion":
            row["promoted_decision"] = "unsafe_memory_answer_that_must_not_be_selected"
    mutated_rows = mod.build_pilot_rows(root=REPO, upstream_artifacts=mutated)

    selected = {
        row.task_id: mod.decision_for_route(row, mod.choose_memory_route(row)) for row in rows
    }
    mutated_selected = {
        row.task_id: mod.decision_for_route(row, mod.choose_memory_route(row))
        for row in mutated_rows
    }

    assert selected == mutated_selected
    assert any(row.memory_feature_active for row in rows)
    assert all(row.memory_answer_injection_blocked for row in rows)
    assert all(
        row["selected_decision_source"] != "memory_promoted_decision"
        for row in mod.evaluate_pilot(rows)["memory_assisted_rows"]
    )


def test_scenario_verify_5276_pilot_preserves_quality_suppresses_memory_and_rolls_back() -> None:
    """SCENARIO-VERIFY-5276: memory allocation avoids calls without unsafe accepts."""

    rows = mod.build_pilot_rows(root=REPO, upstream_artifacts=mod.load_upstream_artifacts(REPO))
    pilot = mod.evaluate_pilot(rows)

    assert pilot["memory_verifier_dose_ready"] is True
    assert pilot["calls_avoided_rate"] == 0.857143
    assert pilot["decision_quality_delta"] == 0.0
    assert pilot["unsafe_false_accepts"] == 0
    assert pilot["rollback_exercised"] is True
    assert pilot["memory_scope_violations_blocked"] == 4
    assert pilot["allocation_changed_by_memory_count"] >= 1
    assert pilot["baseline_metrics"]["always_full"]["quality_rate"] == 1.0
    assert pilot["memory_assisted_metrics"]["quality_rate"] == 1.0
    assert (
        pilot["baseline_metrics"]["no_memory_scheduler"]["full_verifier_calls"]
        > pilot["memory_assisted_metrics"]["full_verifier_calls"]
    )
    assert pilot["route_counts"] == {
        "cheap_deterministic": 2,
        "full_verifier": 1,
        "memory_guided_deterministic_check": 3,
        "no_verifier": 1,
    }


def test_req_verify_5276_edge_branches_fail_closed_and_report_nonready_states(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5276: defensive branches fail closed without promoting metrics."""

    assert mod.extract_model_specs({"MODEL_SPECS": {"value": "bad-shape"}}) == {}
    assert (
        mod.extract_model_specs(
            {
                "MODEL_SPECS": {
                    "value": {
                        "flagship_moe": "bad-row",
                        "flagship_dense": {
                            "hf_id": "unsloth/not-the-mandated-model-GGUF",
                            "status": "local_gguf_resolved",
                        },
                        "middle_moe": {
                            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                            "status": "missing_local_gguf",
                        },
                    }
                }
            }
        )
        == {}
    )

    artifacts = mod.load_upstream_artifacts(REPO)
    artifacts["exp5271"] = {
        **artifacts["exp5271"],
        "telemetry_harness_ready": False,
        "MODEL_SPECS": {"value": {}},
    }
    artifacts["exp5275"] = {**artifacts["exp5275"], "memory_decision_history_ready": False}
    artifacts["exp5264"] = {**artifacts["exp5264"], "scheduler_ready": False}
    preconditions = mod.check_preconditions(root=tmp_path, upstream_artifacts=artifacts)

    assert preconditions["all_gates_ready"] is False
    assert {
        "exp5271.telemetry_harness_ready",
        "exp5275.memory_decision_history_ready",
        "exp5264.scheduler_ready",
        "headline_sota_roles_ready",
        "ops.exclusion_manifest_present",
        "experiment_5276_not_retired",
    } <= set(preconditions["blockers"])
    assert mod._exclusion_manifest_allows(tmp_path) is False
    assert mod._sha256_file(tmp_path / "missing.json") is None

    (tmp_path / "ops").mkdir()
    (tmp_path / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        "retired:\n- experiment_id: 5276\n",
        encoding="utf-8",
    )
    assert mod._exclusion_manifest_allows(tmp_path) is False

    malformed_memory_index = mod._memory_index(
        {
            "governance_rows": [
                None,
                {"task_scope": "", "governance_action": "promote", "active": True},
            ]
        }
    )
    assert malformed_memory_index == {"active_by_scope": {}, "suppressed_by_scope": {}}

    rows = mod.build_pilot_rows(root=REPO, upstream_artifacts=mod.load_upstream_artifacts(REPO))
    fail_closed = replace(rows[0], receipt_complete=False)

    assert mod.choose_memory_route(fail_closed) == mod.ROUTE_FULL
    assert mod.choose_no_memory_route(fail_closed) == mod.ROUTE_FULL
    assert mod.decision_for_route(rows[0], mod.ROUTE_NO_VERIFIER) == rows[0].no_verifier_decision
    assert mod.decision_for_route(rows[0], mod.ROUTE_CHEAP) == rows[0].cheap_decision
    assert mod._is_false_accept("attempt_promotion", "block_until_verified") is True
    assert mod._is_false_accept("attempt_promotion", "use_safe_check") is False

    ready_preconditions = {"all_gates_ready": True, "blockers": []}
    ready_pilot = mod.evaluate_pilot(rows)
    assert "unsafe false accepts" in mod._honest_verdict(
        ready_preconditions,
        {**ready_pilot, "unsafe_false_accepts": 1},
    )
    assert "reduced decision quality" in mod._honest_verdict(
        ready_preconditions,
        {**ready_pilot, "decision_quality_delta": -0.1},
    )
    assert "avoided no full verifier calls" in mod._honest_verdict(
        ready_preconditions,
        {**ready_pilot, "calls_avoided_rate": 0.0},
    )
    assert "did not exercise rollback" in mod._honest_verdict(
        ready_preconditions,
        {**ready_pilot, "rollback_exercised": False},
    )
    assert "did not satisfy all safety gates" in mod._honest_verdict(
        ready_preconditions,
        {**ready_pilot, "memory_verifier_dose_ready": False},
    )


def test_req_verify_5276_artifact_schema_and_run_are_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-5276: run() writes the principle-wrapped result artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    commands_run = [{"command": "unit ready", "outcome": "passed"}]

    artifact = mod.run(root=REPO, result_path=result_path, commands_run=commands_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "positive" in artifact["honest_verdict"]["value"]
    assert artifact["memory_verifier_dose_ready"]["value"] is True
    assert artifact["commands_run"] == commands_run
    assert artifact["duration_s"] >= 10.0
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["source_artifact_checksums"]["exp5271"].startswith("sha256:")
    assert artifact["source_artifact_checksums"]["exp5275"].startswith("sha256:")
    assert artifact["source_artifact_checksums"]["exp5264"].startswith("sha256:")

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert "value" in artifact[field]
        assert "principle" in artifact[field]
    mod.validate_artifact(artifact)


def test_req_verify_5276_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5276: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, commands_run=result["commands_run"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == "live_llm_inference_local_gguf_sota"
    assert result["memory_verifier_dose_ready"]["value"] is True
    assert result["calls_avoided_rate"]["value"] == 0.857143
    assert result["decision_quality_delta"]["value"] == 0.0
    assert result["unsafe_false_accepts"]["value"] == 0
    assert result["rollback_exercised"]["value"] is True
    assert result["memory_scope_violations_blocked"]["value"] == 4
    mod.validate_artifact(result)
