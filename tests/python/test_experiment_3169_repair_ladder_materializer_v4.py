"""Tests for Exp 3169 repair ladder materializer v4.

Spec refs: REQ-VERIFY-3169, SCENARIO-VERIFY-3169.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from carnot.verify import repair_ladder_materializer_v4 as mod


REQUIRED_FIELDS = {
    "repair_ladder_materializer_v4_ready",
    "gated_skip",
    "gated_skip_reason",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "selected_repair_rows",
    "repair_attempt_count",
    "exact_authority_accept_count",
    "repair_success_delta",
    "false_repair_accept_rate",
    "intent_preservation_rate",
    "headline_repair_claim_allowed",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_common_sources(
    root: Path,
    *,
    gate_state: str = "blocked_flagged_verifier",
    selected_rows: list[dict[str, Any]] | None = None,
    usable_model: bool = False,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No fake repair claims\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/experiment_template.py").write_text(
        "cached_sota_pair policy\n", encoding="utf-8"
    )
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3169\nSCENARIO-VERIFY-3169\n"
        "results/experiment_3169_repair_ladder_materializer_v4.json\n",
        encoding="utf-8",
    )
    rows = [] if selected_rows is None else selected_rows
    blockers = [] if gate_state == "unblocked" else ["flagged_adversarial=true"]
    _write_json(
        root,
        mod.EXP3168_REL_PATH,
        {
            "artifact": "experiment_3168_repair_gate_decision_v3",
            "repair_gate_decision_v3_ready": True,
            "repair_gate_state": gate_state,
            "gated_skip": gate_state != "unblocked",
            "gated_skip_reason": "exp3165 preflight failed",
            "repair_blockers": blockers,
            "selected_repair_rows": rows,
            "false_accept_rate": 0.0,
            "false_accept_gate_passed": gate_state == "unblocked",
            "flagged_adversarial": gate_state != "unblocked",
            "controlled_invariance_passed": gate_state == "unblocked",
            "exact_authority_ready": gate_state == "unblocked",
            "headline_claim_allowed": gate_state == "unblocked",
            "inference_substrate": {
                "executes_models": False,
                "live_model_calls": 0,
                "repair_calls": 0,
            },
            "honest_verdict": (
                "complete: repair_gate_state=unblocked"
                if gate_state == "unblocked"
                else f"{gate_state}: blocked fixture"
            ),
        },
    )
    model_path = root / "models" / "gemma26.gguf"
    if usable_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("tiny fixture model path", encoding="utf-8")
    _write_json(
        root,
        mod.EXP3167_REL_PATH,
        {
            "artifact": "experiment_3167_clean_live_sota_verifier_rerun_v9",
            "model_specs": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "name": "Gemma4-26B-A4B-it",
                    "usable_locally": usable_model,
                    "cache_present": usable_model,
                    "model_path": model_path.as_posix() if usable_model else None,
                    "legacy_small_model": False,
                }
            ],
            "selected_model_ids": (
                ["unsloth/gemma-4-26B-A4B-it-GGUF"] if usable_model else []
            ),
            "live_call_count": 0,
            "inference_substrate": {"executes_models": False, "live_model_calls": 0},
        },
    )
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": True,
            "replay_false_accept_rate": 0.0,
        },
    )
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "artifact": "experiment_3138_canonical_answer_vericot_grounding_pilot_v1",
            "canonical_grounding_pilot_v1_ready": True,
            "residual_false_accept_rows": [],
        },
    )
    _write_json(
        root,
        mod.EXP3115_REL_PATH,
        {
            "artifact": "experiment_3115_explicit_repair_gate_micro_panel_v4",
            "repair_success_delta": 0.0,
            "false_repair_accept_rate": 0.0,
            "intent_preservation_rate": 0.0,
        },
    )


def _selected_rows() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "row-a",
            "exact_authority_constraints": {
                "exact_label": "INVALID",
                "expected_action": "reject",
                "exact_safe_decision": "abstain",
                "canonical_decision": "abstain",
                "solver_or_test_authority": "z3_solver",
            },
        },
        {
            "row_id": "row-b",
            "exact_authority_constraints": {
                "exact_label": "UNSAT",
                "expected_action": "reject",
                "exact_safe_decision": "abstain",
                "canonical_decision": "abstain",
                "solver_or_test_authority": "python_test",
            },
        },
    ]


def _stub_repair_runner(
    rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assert [row["row_id"] for row in rows] == ["row-a", "row-b"]
    assert [model["hf_id"] for model in model_specs] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    return [
        {
            "row_id": "row-a",
            "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "accepted": True,
            "exact_verified": True,
            "canonical_grounded": True,
            "controlled_invariance_passed": True,
            "monitor_replay_passed": True,
            "intent_preserved": True,
        },
        {
            "row_id": "row-b",
            "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "accepted": False,
            "exact_verified": False,
            "canonical_grounded": True,
            "controlled_invariance_passed": True,
            "monitor_replay_passed": True,
            "intent_preserved": True,
        },
    ]


def test_req_verify_3169_spec_anchor_exists() -> None:
    """REQ-VERIFY-3169: OpenSpec declares the repair ladder materializer."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3169" in spec
    assert "SCENARIO-VERIFY-3169" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_ladder_materializer_v4_ready" in spec
    assert "headline_repair_claim_allowed" in spec


def test_scenario_verify_3169_blocked_gate_writes_full_no_call_skip(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3169: a blocked Exp 3168 gate forbids repair generation."""

    _write_common_sources(tmp_path, gate_state="blocked_flagged_verifier")

    def fail_if_called(
        rows: Sequence[Mapping[str, Any]],
        model_specs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        raise AssertionError("repair runner must not be called when Exp 3168 blocks")

    output_path = mod.write_artifact(
        tmp_path,
        repair_runner=fail_if_called,
        started_s=4.0,
        now_s=6.25,
        tests_run=["focused-3169"],
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_ladder_materializer_v4_ready"] is True
    assert artifact["gated_skip"] is True
    assert artifact["gated_skip_reason"].startswith(
        "repair gate blocked: blocked_flagged_verifier"
    )
    assert artifact["selected_model_ids"] == []
    assert artifact["live_call_count"] == 0
    assert artifact["selected_repair_rows"] == []
    assert artifact["repair_attempt_count"] == 0
    assert artifact["exact_authority_accept_count"] == 0
    assert artifact["repair_success_delta"] == pytest.approx(0.0)
    assert artifact["false_repair_accept_rate"] == pytest.approx(0.0)
    assert artifact["intent_preservation_rate"] == pytest.approx(0.0)
    assert artifact["headline_repair_claim_allowed"] is False
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["tests_run"] == ["focused-3169"]
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["honest_verdict"].startswith("blocked_repair_gate:")
    assert all("legacy" not in model["hf_id"] for model in artifact["model_specs"])


def test_scenario_verify_3169_missing_gate_is_complete_skip(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3169: missing Exp 3168 evidence becomes an explicit skip."""

    (tmp_path / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["repair_ladder_materializer_v4_ready"] is True
    assert artifact["gated_skip"] is True
    assert artifact["gated_skip_reason"] == "repair gate decision artifact is missing"
    assert artifact["live_call_count"] == 0
    assert artifact["source_artifacts"][5]["path"] == mod.EXP3168_REL_PATH.as_posix()
    assert artifact["source_artifacts"][5]["present"] is False
    assert artifact["honest_verdict"].startswith("blocked_repair_gate:")


def test_req_verify_3169_unblocked_gate_without_usable_model_still_skips(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3169: unblocked gates also require mandated local SOTA models."""

    _write_common_sources(
        tmp_path,
        gate_state="unblocked",
        selected_rows=_selected_rows(),
        usable_model=False,
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["gated_skip"] is True
    assert artifact["gated_skip_reason"] == "no mandated local SOTA GGUF model is usable"
    assert artifact["selected_repair_rows"] == _selected_rows()
    assert artifact["selected_model_ids"] == []
    assert artifact["repair_attempt_count"] == 0
    assert artifact["headline_repair_claim_allowed"] is False
    assert artifact["honest_verdict"].startswith("blocked_repair_runtime:")


def test_scenario_verify_3169_unblocked_stubbed_panel_scores_exact_accepts(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3169: accepted repairs require exact and monitor evidence."""

    _write_common_sources(
        tmp_path,
        gate_state="unblocked",
        selected_rows=_selected_rows(),
        usable_model=True,
    )

    artifact = mod.build_artifact(
        tmp_path,
        repair_runner=_stub_repair_runner,
        started_s=10.0,
        now_s=13.0,
        tests_run=["focused-run"],
    )

    assert artifact["gated_skip"] is False
    assert artifact["gated_skip_reason"] == ""
    assert artifact["selected_model_ids"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["live_call_count"] == 2
    assert artifact["repair_attempt_count"] == 2
    assert artifact["exact_authority_accept_count"] == 1
    assert artifact["repair_success_delta"] == pytest.approx(0.5)
    assert artifact["false_repair_accept_rate"] == pytest.approx(0.0)
    assert artifact["intent_preservation_rate"] == pytest.approx(1.0)
    assert artifact["headline_repair_claim_allowed"] is True
    assert artifact["inference_substrate"]["executes_models"] is True
    assert artifact["inference_substrate"]["repair_runner_kind"] == "injected_repair_runner"
    assert artifact["repair_attempts"][0]["accepted_by_exact_authority"] is True
    assert artifact["repair_attempts"][1]["accepted_by_exact_authority"] is False
    assert artifact["duration_s"] == pytest.approx(3.0)
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3169_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3169: helper paths fail closed and validation rejects unsafe shapes."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    missing_path = tmp_path / "missing.json"

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(missing_path) == {}
    assert mod.mapping_rows([{"a": 1}, [], {"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert mod.mapping_rows({"not": "list"}) == []
    assert mod.sha256_file(missing_path) is None
    assert mod.duration(5.0, 3.0) == pytest.approx(0.0)
    assert mod.rate(1, 0) == pytest.approx(0.0)
    assert mod.row_id_from({"fixture_id": "fixture-a"}) == "fixture-a"

    specs = mod.model_specs_from_sources(
        {
            "model_specs": [
                {
                    "hf_id": "legacy/small",
                    "usable_locally": True,
                },
                {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "usable_locally": True,
                    "model_path": "/tmp/qwen.gguf",
                },
            ]
        },
        {
            "model_specs": [
                {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "cache_present": True,
                }
            ]
        },
    )
    assert [model["hf_id"] for model in specs] == list(mod.MANDATED_MODEL_IDS)
    assert mod.usable_mandated_model_specs(specs)[0]["hf_id"] == (
        "unsloth/Qwen3.6-35B-A3B-GGUF"
    )
    assert mod.usable_mandated_model_specs([{"hf_id": "legacy/small", "selected": True}]) == []
    assert (
        mod.repair_run_decision(
            gate_present=True,
            gate={"repair_gate_decision_v3_ready": False},
            selected_rows=[],
            usable_specs=[],
            repair_runner=None,
        )
        == (False, "repair gate decision artifact is not ready")
    )
    assert (
        mod.repair_run_decision(
            gate_present=True,
            gate={"repair_gate_decision_v3_ready": True, "repair_gate_state": "unblocked"},
            selected_rows=[],
            usable_specs=specs[:1],
            repair_runner=_stub_repair_runner,
        )
        == (False, "repair gate unblocked but selected_repair_rows is empty")
    )
    assert (
        mod.repair_run_decision(
            gate_present=True,
            gate={"repair_gate_decision_v3_ready": True, "repair_gate_state": "unblocked"},
            selected_rows=_selected_rows(),
            usable_specs=specs[:1],
            repair_runner=None,
        )
        == (False, "live repair runner is not configured")
    )
    assert mod.first_gate_blocker({"gated_skip_reason": "gate skipped"}) == "gate skipped"

    metrics = mod.repair_metrics(
        [
            {
                "accepted": True,
                "exact_verified": False,
                "canonical_grounded": True,
                "controlled_invariance_passed": True,
                "monitor_replay_passed": True,
                "intent_preserved": False,
            },
            {
                "accepted": True,
                "exact_verified": True,
                "canonical_grounded": True,
                "controlled_invariance_passed": True,
                "monitor_replay_passed": True,
                "intent_preserved": True,
            },
        ],
        selected_count=4,
    )
    assert metrics["exact_authority_accept_count"] == 1
    assert metrics["repair_success_delta"] == pytest.approx(0.25)
    assert metrics["false_repair_accept_rate"] == pytest.approx(0.5)
    assert metrics["intent_preservation_rate"] == pytest.approx(0.5)

    _write_common_sources(
        tmp_path,
        gate_state="unblocked",
        selected_rows=_selected_rows(),
        usable_model=True,
    )

    def raising_runner(
        rows: Sequence[Mapping[str, Any]],
        model_specs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        raise RuntimeError("unit boom")

    artifact = mod.build_artifact(tmp_path, repair_runner=raising_runner)
    assert artifact["gated_skip"] is False
    assert artifact["repair_attempt_count"] == 1
    assert artifact["repair_attempts"][0]["verification_errors"] == [
        "repair_runner_error: RuntimeError: unit boom"
    ]
    assert artifact["headline_repair_claim_allowed"] is False

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="finite rate"):
        mod.validate_artifact(artifact | {"repair_success_delta": float("nan")})
    with pytest.raises(ValueError, match="selected_model_ids"):
        mod.validate_artifact(artifact | {"selected_model_ids": ["legacy/small"]})
    with pytest.raises(ValueError, match="gated skip"):
        mod.validate_artifact(artifact | {"gated_skip": True, "live_call_count": 1})
    with pytest.raises(ValueError, match="executed repair"):
        mod.validate_artifact(artifact | {"gated_skip": False, "repair_attempt_count": 0})
    with pytest.raises(ValueError, match="headline"):
        mod.validate_artifact(artifact | {"headline_repair_claim_allowed": True})
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked_after_run"})
