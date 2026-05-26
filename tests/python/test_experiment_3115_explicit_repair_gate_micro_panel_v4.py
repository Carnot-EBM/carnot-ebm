"""Tests for Exp 3115 explicit repair gate and micro-panel v4.

Spec refs: REQ-VERIFY-3115, SCENARIO-VERIFY-3115.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import explicit_repair_gate_micro_panel_v4 as exp


REQUIRED_FIELDS = {
    "repair_micro_panel_v4_artifact_ready",
    "repair_unblocked",
    "repair_run_executed",
    "gate_block_reason",
    "model_specs",
    "selected_headline_model_ids",
    "exact_ground_truth_count",
    "repair_success_delta",
    "false_repair_accept_rate",
    "intent_preservation_rate",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _repair_targets() -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": "arith-invalid",
            "fragment_id": "arith-invalid:assert_claim",
            "failing_constraint": "claimed_value == computed_value",
            "expected_direction": "replace claimed value 16 with 14",
            "solver_evidence": {
                "authority": "python_ast_literal_evaluator",
                "assertion": "assert (((4 + 3) * 2) - 0) == 16",
                "expression": "(4 + 3) * 2 - 0",
                "claimed_fragment": "16",
                "claimed_value": 16,
                "computed_value": 14,
            },
        },
        {
            "fixture_id": "json-repair",
            "fragment_id": "json-repair:json_document",
            "failing_constraint": "valid_json_document",
            "expected_direction": "produce parseable JSON while preserving required fields",
            "solver_evidence": {
                "authority": "python_json_parser",
                "candidate": '{"mode": "bounded" "limit": 2}',
                "parse_error": "Expecting ',' delimiter",
            },
        },
        {
            "fixture_id": "numeric-repair",
            "fragment_id": "numeric-repair:constraint:2",
            "failing_constraint": "rx_1 + ry_1 == 11",
            "expected_direction": "increase sum by 8 across ['rx_1', 'ry_1']",
            "solver_evidence": {
                "authority": "deterministic_integer_constraint_evaluator",
                "assignment": {"rx_1": 2, "ry_1": 1},
                "constraint": "rx_1 + ry_1 == 11",
                "lhs_value": 3,
                "rhs_value": 11,
            },
        },
        {
            "fixture_id": "py-repair",
            "fragment_id": "py-repair:assert_claim",
            "failing_constraint": "claimed_value == computed_value",
            "expected_direction": "replace claimed value 13 with 12",
            "solver_evidence": {
                "authority": "python_ast_literal_evaluator",
                "assertion": "assert ((7 * 2) - 2) == 13",
                "expression": "7 * 2 - 2",
                "claimed_fragment": "13",
                "claimed_value": 13,
                "computed_value": 12,
            },
        },
    ]


def _write_sources(
    root: Path,
    *,
    gate_state: str = "unblocked",
    selected_model: bool = True,
    target_rows: list[dict[str, Any]] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No fake repair results\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/experiment_template.py").write_text(
        "cached_sota_pair policy\n", encoding="utf-8"
    )
    _write_json(
        root,
        exp.EXP3097_REL_PATH,
        {
            "artifact": "experiment_3097_exact_fixture_eval_protocol_audit_v1",
            "eval_protocol_ready": True,
            "usable_fixture_count": 72,
            "honest_verdict": "complete: eval_protocol_ready=true",
        },
    )
    _write_json(
        root,
        exp.EXP3110_REL_PATH,
        {
            "artifact": "experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1",
            "sota_model_manifest_ready": True,
            "mandatory_headline_model_ids": list(exp.MANDATORY_MODEL_IDS),
            "selected_headline_model_ids": (
                ["unsloth/gemma-4-26B-A4B-it-GGUF"] if selected_model else []
            ),
            "honest_verdict": "complete: sota_model_manifest_ready=true",
        },
    )
    model_path = root / "models" / "selected.gguf"
    if selected_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("tiny gguf fixture", encoding="utf-8")
    model_specs = [
        {
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "name": "Gemma4-26B-A4B-it",
            "cache_present": selected_model,
            "cache_status": "cached" if selected_model else "cache_missing",
            "selected": selected_model,
            "model_path": model_path.as_posix() if selected_model else None,
            "file_size_bytes": 17,
        }
    ]
    _write_json(
        root,
        exp.EXP3113_REL_PATH,
        {
            "artifact": "experiment_3113_diagnostic_local_sota_verifier_calibration_v5",
            "diagnostic_verifier_calibration_v5_ready": True,
            "repair_gate_state": gate_state,
            "exact_ground_truth_count": 27,
            "model_specs": model_specs,
            "selected_headline_model_ids": (
                ["unsloth/gemma-4-26B-A4B-it-GGUF"] if selected_model else []
            ),
            "exp3115_repair_gate_explanation": {
                "repair_gate_state": gate_state,
                "downstream_action": "repair_gate_unblocked",
            },
            "honest_verdict": f"complete: repair_gate_state={gate_state}",
        },
    )
    rows = _repair_targets() if target_rows is None else target_rows
    _write_jsonl(root, exp.REPAIR_TARGET_MANIFEST_REL_PATH, rows)
    _write_json(
        root,
        exp.EXP3114_REL_PATH,
        {
            "artifact": "experiment_3114_fragment_level_code_constraint_verification_pilot_v1",
            "fragment_verification_pilot_ready": True,
            "repair_target_manifest_path": exp.REPAIR_TARGET_MANIFEST_REL_PATH.as_posix(),
            "failing_fragment_count": len(rows),
            "honest_verdict": "complete: fragment verification pilot ready",
        },
    )


def _stub_repair_generator(
    prompt: str,
    target: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> str:
    assert "REQ-VERIFY-3115" in prompt
    assert model_spec["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    fragment_id = str(target["fragment_id"])
    repairs = {
        "arith-invalid:assert_claim": ' {"repaired_fragment": "assert (((4 + 3) * 2) - 0) == 14"} ',
        "json-repair:json_document": '{"repaired_fragment": "{\\"mode\\": \\"bounded\\", \\"limit\\": 2}"}',
        "numeric-repair:constraint:2": '{"repaired_fragment": "{\\"rx_1\\": 10, \\"ry_1\\": 1}"}',
        "py-repair:assert_claim": "assert ((7 * 2) - 2) == 12",
    }
    return repairs[fragment_id]


def test_req_verify_3115_spec_anchor_exists() -> None:
    """REQ-VERIFY-3115: OpenSpec declares the explicit repair boundary artifact."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3115" in spec
    assert "SCENARIO-VERIFY-3115" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_run_executed" in spec
    assert "repair_success_delta" in spec


def test_scenario_verify_3115_blocked_gate_writes_boundary_without_generation(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3115: blocked upstream gates still produce a terminal artifact."""

    _write_sources(tmp_path, gate_state="blocked_negative_delta")

    def fail_if_called(
        prompt: str,
        target: Mapping[str, Any],
        model_spec: Mapping[str, Any],
    ) -> str:
        raise AssertionError("repair generation must not run for a blocked gate")

    output_path = exp.write_artifact(
        tmp_path,
        repair_generator=fail_if_called,
        started_s=10.0,
        now_s=12.5,
        tests_run=["focused-blocked"],
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_micro_panel_v4_artifact_ready"] is True
    assert artifact["repair_unblocked"] is False
    assert artifact["repair_run_executed"] is False
    assert artifact["gate_block_reason"].startswith(
        "exp3113 repair_gate_state=blocked_negative_delta"
    )
    assert artifact["exact_ground_truth_count"] == 4
    assert artifact["repair_success_delta"] == pytest.approx(0.0)
    assert artifact["false_repair_accept_rate"] == pytest.approx(0.0)
    assert artifact["intent_preservation_rate"] == pytest.approx(0.0)
    assert artifact["tests_run"] == ["focused-blocked"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["honest_verdict"].startswith("blocked_repair_gate:")
    exp.validate_artifact(artifact)


def test_scenario_verify_3115_unblocked_runs_stubbed_exact_repair_panel(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3115: executed repairs are checked by exact authorities."""

    _write_sources(tmp_path, gate_state="unblocked")

    artifact = exp.build_artifact(
        tmp_path,
        repair_generator=_stub_repair_generator,
        started_s=20.0,
        now_s=23.0,
        tests_run=["focused-executed"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_micro_panel_v4_artifact_ready"] is True
    assert artifact["repair_unblocked"] is True
    assert artifact["repair_run_executed"] is True
    assert artifact["gate_block_reason"] == "repair_gate_unblocked"
    assert artifact["exact_ground_truth_count"] == 4
    assert artifact["repair_success_delta"] == pytest.approx(1.0)
    assert artifact["false_repair_accept_rate"] == pytest.approx(0.0)
    assert artifact["intent_preservation_rate"] == pytest.approx(1.0)
    assert artifact["selected_headline_model_ids"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"][0]["selected"] is True
    assert artifact["inference_substrate"]["executes_models"] is True
    assert artifact["inference_substrate"]["repair_generator_kind"] == "injected_repair_generator"
    assert artifact["tests_run"] == ["focused-executed"]
    assert artifact["duration_s"] == pytest.approx(3.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(artifact["repair_rows"]) == 4
    assert all(row["accepted"] for row in artifact["repair_rows"])
    assert all(row["exact_verified"] for row in artifact["repair_rows"])
    assert all(row["intent_preserved"] for row in artifact["repair_rows"])
    assert all(row["exists"] for row in artifact["source_artifacts"])
    exp.validate_artifact(artifact)


def test_req_verify_3115_runtime_blocks_when_unblocked_but_model_or_targets_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3115: unblocked gates still fail closed on missing runtime inputs."""

    _write_sources(tmp_path, selected_model=False)
    missing_model = exp.build_artifact(tmp_path, started_s=1.0, now_s=2.0)

    assert missing_model["repair_unblocked"] is True
    assert missing_model["repair_run_executed"] is False
    assert missing_model["gate_block_reason"].startswith("missing_selected_mandated_sota_model")
    assert missing_model["honest_verdict"].startswith("blocked_repair_runtime:")

    targetless_root = tmp_path / "targetless"
    _write_sources(targetless_root, target_rows=[])
    no_targets = exp.build_artifact(
        targetless_root,
        repair_generator=_stub_repair_generator,
        started_s=1.0,
        now_s=2.0,
    )

    assert no_targets["repair_unblocked"] is True
    assert no_targets["repair_run_executed"] is False
    assert no_targets["gate_block_reason"].startswith("missing_repair_targets")
    assert no_targets["exact_ground_truth_count"] == 0
    exp.validate_artifact(no_targets)


def test_req_verify_3115_verification_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3115: helper paths keep exact checks and validation deterministic."""

    target = _repair_targets()[0]
    ok = exp.verify_repair_candidate(target, "assert (((4 + 3) * 2) - 0) == 14")
    wrong_intent = exp.verify_repair_candidate(target, "assert 14 == 14")
    malformed_json = exp.verify_repair_candidate(_repair_targets()[1], '{"mode": "bounded"')
    json_list = exp.verify_repair_candidate(_repair_targets()[1], "[]")
    json_missing_key = exp.verify_repair_candidate(_repair_targets()[1], '{"mode": "bounded"}')
    bad_numeric = exp.verify_repair_candidate(_repair_targets()[2], '{"rx_1": 2, "ry_1": 1}')
    numeric_not_object = exp.verify_repair_candidate(_repair_targets()[2], "[]")
    numeric_changed_keys = exp.verify_repair_candidate(
        _repair_targets()[2],
        '{"rx_1": 10, "ry_1": 1, "extra": 0}',
    )
    wrong_assert_value = exp.verify_repair_candidate(
        target,
        "assert (((4 + 3) * 2) - 0) == 15",
    )
    non_assertion = exp.verify_assertion_repair(
        target["solver_evidence"],
        "value = 14",
    )
    multi_compare = exp.verify_assertion_repair(
        target["solver_evidence"],
        "assert 1 < 2 < 3",
    )
    unknown = exp.verify_repair_candidate(
        {
            "fixture_id": "u",
            "fragment_id": "u:unknown",
            "solver_evidence": {"authority": "unknown"},
        },
        "anything",
    )

    assert exp.read_json_object(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert exp.read_json_object(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(list_json) == {}
    malformed_jsonl = tmp_path / "rows.jsonl"
    malformed_jsonl.write_text('\nnot-json\n{"ok": true}\n[]\n', encoding="utf-8")
    assert exp.read_jsonl_rows(malformed_jsonl) == [{"ok": True}]
    assert exp.read_jsonl_rows(tmp_path / "missing.jsonl") == []
    assert exp.extract_repaired_fragment('{"repaired_fragment": "x"}') == "x"
    assert exp.extract_repaired_fragment("```python\nassert 1 == 1\n```") == "assert 1 == 1"
    assert exp.rate(1, 0) == 0.0
    assert exp.duration(5.0, 3.0) == pytest.approx(0.0)
    assert exp.sha256_file(tmp_path / "absent") is None
    assert exp.assertion_left_ast("value = 14") == ""
    assert ok["exact_verified"] is True
    assert ok["intent_preserved"] is True
    assert wrong_intent["exact_verified"] is True
    assert wrong_intent["intent_preserved"] is False
    assert wrong_assert_value["exact_verified"] is False
    assert wrong_assert_value["intent_preserved"] is True
    assert wrong_assert_value["verification_errors"] == ["assertion_not_exact"]
    assert non_assertion["exact_verified"] is False
    assert "candidate is not a simple assertion" in non_assertion["verification_errors"][0]
    assert multi_compare["exact_verified"] is False
    assert "multiple comparators" in multi_compare["verification_errors"][0]
    assert malformed_json["exact_verified"] is False
    assert "json_parse_error" in malformed_json["verification_errors"][0]
    assert json_list["exact_verified"] is False
    assert json_list["verification_errors"] == [
        "json_repair_not_object",
        "json_required_keys_not_preserved",
    ]
    assert json_missing_key["exact_verified"] is True
    assert json_missing_key["intent_preserved"] is False
    assert bad_numeric["exact_verified"] is False
    assert bad_numeric["intent_preserved"] is True
    assert numeric_not_object["exact_verified"] is False
    assert "numeric repair must be a JSON object" in numeric_not_object["verification_errors"][0]
    assert numeric_changed_keys["exact_verified"] is True
    assert numeric_changed_keys["intent_preserved"] is False
    assert unknown["verification_errors"] == ["unsupported repair authority: unknown"]
    assert (
        exp.selected_cached_model_specs(
            [
                {
                    "hf_id": "legacy/small-model",
                    "selected": True,
                    "cache_present": True,
                    "model_path": "/tmp/legacy.gguf",
                },
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "selected": False,
                    "cache_present": True,
                    "model_path": "/tmp/not-selected.gguf",
                },
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "selected": True,
                    "cache_present": True,
                },
            ],
            [],
        )
        == []
    )
    assert exp.runtime_block_reason(
        _repair_targets(),
        [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "model_path": None}],
        require_real_model_file=True,
    ).startswith("missing_selected_model_path")
    assert exp.runtime_block_reason(
        _repair_targets(),
        [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "model_path": tmp_path / "missing.gguf"}],
        require_real_model_file=True,
    ).startswith("missing_selected_model_path")
    tiny_model = tmp_path / "tiny.gguf"
    tiny_model.write_text("tiny", encoding="utf-8")
    assert exp.runtime_block_reason(
        _repair_targets(),
        [{"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "model_path": tiny_model.as_posix()}],
        require_real_model_file=True,
    ).startswith("unusable_selected_model_path")

    def raising_generator(
        prompt: str,
        repair_target: Mapping[str, Any],
        model_spec: Mapping[str, Any],
    ) -> str:
        raise RuntimeError("unit boom")

    generated_error_rows = exp.run_repair_panel(
        _repair_targets()[:1],
        {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "model_path": tiny_model.as_posix()},
        raising_generator,
    )
    assert generated_error_rows[0]["accepted"] is False
    assert (
        "generation_error: RuntimeError: unit boom"
        in generated_error_rows[0]["verification_errors"]
    )

    _write_sources(tmp_path)
    artifact = exp.build_artifact(tmp_path, repair_generator=_stub_repair_generator)
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="finite metric"):
        exp.validate_artifact(artifact | {"repair_success_delta": float("nan")})
    with pytest.raises(ValueError, match="cannot execute"):
        exp.validate_artifact(artifact | {"repair_unblocked": False})
    with pytest.raises(ValueError, match="gate_block_reason"):
        exp.validate_artifact(artifact | {"gate_block_reason": ""})
    with pytest.raises(ValueError, match="success prefix"):
        exp.validate_artifact(artifact | {"honest_verdict": "blocked_after_execution"})
    with pytest.raises(ValueError, match="blocked verdict"):
        blocked = artifact | {"repair_run_executed": False, "honest_verdict": "complete: bad"}
        exp.validate_artifact(blocked)
